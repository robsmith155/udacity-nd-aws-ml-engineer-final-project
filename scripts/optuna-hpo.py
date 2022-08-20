"""Script to run Optuna hyperparameter search.

This will run a TPE hyperparameter search for the selected model. 

To view the help for this script, please run `python optuna-hpo.py --help` from the command line.

Example usage:

python optuna-hpo.py --max-epochs 30 --num-trials 100 --no-fast-mode --model-type unet

This will run 100 trials of a UNet network, each run for up to 30 epochs. 

"""

import logging
import os
import pathlib
import sys
from typing import List, Optional, Union

import git
import monai
import pytorch_lightning as pl
import torch
import typer
import yaml
from monai.data import CacheDataset
from optuna.integration import PyTorchLightningPruningCallback

import optuna
import wandb

# Resolve repo root path and add to path
git_repo = git.Repo(".", search_parent_directories=True)
PROJECT_ROOT_PATH = git_repo.working_dir

with open(f"{PROJECT_ROOT_PATH}/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DATA_ROOT_PATH = os.path.join(PROJECT_ROOT_PATH, config["DATA_DIR"])
sys.path.append(PROJECT_ROOT_PATH)

# Import functions from repo modules
from sagemaker_src.brain_datamodule import (
    BrainMRIDataOptuna,
    extract_input_filepaths,
    monai_transformations,
)
from sagemaker_src.brain_model import BrainMRIModel, BrainSegPredictionLogger
from src.data import create_kaggle_token_file, setup_brain_mri_dataset
from src.utils import generate_wandb_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def objective(
    trial: optuna.trial.Trial,
    train_data_dir: Union[str, pathlib.Path],
    val_data_dir: Union[str, pathlib.Path],
    train_transforms: monai.transforms.Compose,
    val_transforms: monai.transforms.Compose,
    train_files: List[dict],
    val_files: List[dict],
    train_ds,
    val_ds,
    model_type: Optional[str] = "unet",
    max_epochs: Optional[int] = 50,
    fast_mode: Optional[bool] = False,
    seed: Optional[int] = 155,
    cache_rate: Optional[float] = 1.0,
    num_workers: Optional[int] = 8,
    study_name: Optional[str] = "optuna-hpo",
    wandb_tracking: Optional[bool] = True,
    precision: Optional[int] = 16,
) -> float:

    torch.cuda.empty_cache()
    if wandb_tracking:
        wandb.finish()  # In case previous run failed to complete successfully

    pl.utilities.seed.seed_everything(seed=seed)

    num_filters_block_1 = trial.suggest_categorical(
        "num_filters_block_1", [8, 16, 32]
    )
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    learning_rate = trial.suggest_loguniform("learning_rate", 0.000001, 0.1)

    # Create Data Module using inherited class made for Optuna (we can pass loaded Datasets to it to prevent loading on each trial which can cause memory issues)
    brain_dm = BrainMRIDataOptuna(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        cache_rate=cache_rate,
        num_workers=num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=batch_size,
        fast_mode=fast_mode,
        train_files=train_files,
        val_files=val_files,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model = BrainMRIModel(
        model_type=model_type,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_filters_block_1=num_filters_block_1,
        dropout_rate=dropout_rate,
        wandb_tracking=wandb_tracking,
    )

    # Save best model based on validation Dice score
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_mean_dice",
        mode="max",
        dirpath=f"{PROJECT_ROOT_PATH}/optuna/{study_name}/trial_{trial.number}",
        filename="model",
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="min",
        check_finite=True,
    )

    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    run_name = f"{study_name}_trial-{trial.number}_{model_type}_bs-{batch_size}_lr-{learning_rate: .5f}_dropout-{dropout_rate: .2f}_filts1-{num_filters_block_1}"

    if wandb_tracking:
        # Setup PyTorch Lightning Trainer object with Weights & Biases tracking
        wandb_logger = pl.loggers.WandbLogger(
            project="brain-mri-segmentation", log_model=False, name=run_name
        )
        hyperparameters = {
            "model_type": model_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "num_filters_block_1": num_filters_block_1,
        }
        wandb_logger.experiment.config.update(hyperparameters)

        log_predictions_callback = BrainSegPredictionLogger()
        wandb_logger.watch(model, log_freq=1000)

        trainer = pl.Trainer(
            precision=precision,
            accelerator=accelerator,
            devices=1,
            logger=wandb_logger,
            max_epochs=max_epochs,
            callbacks=[
                PyTorchLightningPruningCallback(
                    trial, monitor="val_mean_dice"
                ),
                checkpoint_callback,
                early_stop_callback,
                log_predictions_callback,
                swa_callback,
                lr_callback,
            ],
            gradient_clip_val=0.5,
        )

        trial.set_user_attr("wandb_experiment_id", wandb_logger.experiment.id)
    else:
        # Setup PyTorch Lightning Trainer object without Weights & Biases tracking
        trainer = pl.Trainer(
            precision=precision,
            accelerator=accelerator,
            devices=1,
            max_epochs=max_epochs,
            callbacks=[
                PyTorchLightningPruningCallback(
                    trial, monitor="val_mean_dice"
                ),
                checkpoint_callback,
                early_stop_callback,
                swa_callback,
            ],
            gradient_clip_val=0.5,
        )

    trainer.fit(model=model, datamodule=brain_dm)
    max_val_dice = max(model.val_dice_all)

    del brain_dm, trainer, model

    return max_val_dice


# Based on https://optuna.readthedocs.io/en/stable/faq.html#how-to-save-machine-learning-models-trained-in-objective-functions
def wandb_model_artifact_callback(study, frozen_trial):

    if frozen_trial.value >= study.best_value:
        logger.info(
            f"INFO: Current trial score of {frozen_trial.value} is better than all previous trials. Saving model checkpoint to Weights and Biases..."
        )
        model_path = f"{PROJECT_ROOT_PATH}/optuna/{study.study_name}/trial_{frozen_trial.number}/model.ckpt"
        artifact = wandb.Artifact(
            name=f"model-{frozen_trial.user_attrs['wandb_experiment_id']}",
            type="model",
        )
        artifact.add_file(model_path, name="model.ckpt")
        wandb.log_artifact(artifact)
    else:
        logger.info(
            f"INFO: Current trial score of {frozen_trial.value} is worse than best score {study.best_value}. Skipping saving model to W&B."
        )


def run_optuna_hpo(
    model_type: Optional[str] = typer.Option(
        "unet",
        help="Type of model to train, Valid inputs are 'unet' or 'unet-attention'",
    ),
    max_epochs: Optional[int] = typer.Option(
        50, help="Maximum number of epochs to train eac trial"
    ),
    fast_mode: Optional[bool] = typer.Option(
        False,
        help="Whether to run training in fast mode (i.e. all data moved to GPU memory).",
    ),
    num_trials: Optional[int] = typer.Option(
        100,
        help="Maximum number of models with different hyperparameters (trials) to train",
    ),
    seed: Optional[int] = typer.Option(
        155, help="Seed value for experiments."
    ),
    cache_rate: Optional[float] = typer.Option(
        1.0,
        help="Proportion of data to move to memory, with 1.0 being all the data. This helps speed up training",
    ),
    num_workers: Optional[int] = typer.Option(
        8,
        help="Number of workers to use for PyTorch dataloaders. Note that if fast-mode is used, this will be set to 0",
    ),
    resume_search: Optional[bool] = typer.Option(
        True,
        help="If study_name already exists, whether to continue that study.",
    ),
    study_name: Optional[str] = typer.Option(
        "optuna-hpo",
        help="Name of Optuna study. A file named <study_name>.db will be created.",
    ),
    wandb_tracking: Optional[bool] = typer.Option(
        True,
        help="Whether to track the experiments with Weights and Biases. For this you must have saved your API token secret in the same folder as this script.",
    ),
    precision: Optional[int] = typer.Option(
        32,
        help="Precision used for training. Acceptable inputs are 16 or 32 bit training. Note that 16-bit training can speed things up but may be more unstable.",
    ),
):
    # PREPARE DATASET
    create_kaggle_token_file()
    setup_brain_mri_dataset(data_root_path=DATA_ROOT_PATH)
    if wandb_tracking:
        generate_wandb_api_key()

    # CREATE DATASETS
    train_data_dir = f"{DATA_ROOT_PATH}/brain-mri-dataset/train"
    val_data_dir = f"{DATA_ROOT_PATH}/brain-mri-dataset/val"
    train_folders = [f.path for f in os.scandir(train_data_dir) if f.is_dir()]
    val_folders = [f.path for f in os.scandir(val_data_dir) if f.is_dir()]

    train_files = extract_input_filepaths(train_folders)
    val_files = extract_input_filepaths(val_folders)

    train_transforms, val_transforms = monai_transformations(
        fast_mode=fast_mode
    )

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
        copy_cache=False,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
        copy_cache=False,
    )

    # OPTUNA HPO SEARCH
    # Add stream handler of stdout to show the messages (from Optuna guide)
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout)
    )
    storage_name = f"sqlite:///{PROJECT_ROOT_PATH}/optuna/{study_name}.db"

    func = lambda trial: objective(
        trial,
        train_data_dir,
        val_data_dir,
        train_transforms,
        val_transforms,
        train_files,
        val_files,
        train_ds,
        val_ds,
        model_type,
        max_epochs,
        fast_mode,
        seed,
        cache_rate,
        num_workers,
        study_name,
        wandb_tracking,
        precision,
    )
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=10, n_startup_trials=10
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=sampler,
        direction="maximize",
        pruner=pruner,
        load_if_exists=resume_search,
    )
    if wandb_tracking:
        study.optimize(
            func,
            n_trials=num_trials,
            callbacks=[wandb_model_artifact_callback],
        )
    else:
        study.optimize(func, n_trials=num_trials)


if __name__ == "__main__":
    typer.run(run_optuna_hpo)
