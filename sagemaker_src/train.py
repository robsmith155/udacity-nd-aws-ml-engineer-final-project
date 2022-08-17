""" Train PyTorch model on SageMaker container.

This script will train a Monai segmentation model on a Sagemaker PyTorch container. 

"""
import argparse
import logging
import os

import pytorch_lightning as pl
import torch
from brain_datamodule import create_data_module
from brain_model import BrainMRIModel, BrainSegPredictionLogger
from inference import (  # Needed if you deploy directly from a SageMaker PyTorch Estimator
    model_fn,
    predict_fn,
)

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Function below copied from : https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py
def str2bool(v: str):
    """Converts an input string argument to a Boolean.

    This is needed because ArgParse cannot pass Boolean arguments. Therefore, this function converts certain string values to a Boolean (e.g. 'false' converted to the boolean False).

    Args:
        v (str): Input argument

    Raises:
        argparse.ArgumentTypeError: If the passed argument is not a Boolean or one of ("yes", "true", "t", "y", "1") or ("no", "false", "f", "n", "0") then an error is raised.

    Returns:
        bool: Either True or False depending on the input.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):
    """Trains a segmentation model for the Brain MRI dataset.

    Args:
        args (argparse.ArgumentParser): Dictionary of arguments from argparse.
    """
    pl.utilities.seed.seed_everything(seed=args.seed)

    if args.wandb_tracking:
        wandb.finish()
        if "secrets.env" not in os.listdir("."):
            logger.warning(
                "WARNING: wandb_tracking set to True but secrets file not found. Need to obtain API key from https://wandb.ai/authorize"
            )
            wandb.login()
            wandb.sagemaker_auth(path=".")
            logger.info("INFO: W&B secret key file added.")
        else:
            logger.info(
                "INFO: W&B secrets file already exists. Skipping step."
            )

    brain_dm = create_data_module(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        fast_mode=args.fast_mode,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    model = BrainMRIModel(
        model_type=args.model_type,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_filters_block_1=args.num_filters_block_1,
        dropout_rate=args.dropout_rate,
    )

    # saves best model based on validation Dice score
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_mean_dice",
        mode="max",
        dirpath=args.model_dir,
        filename="model",
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="max",
        check_finite=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    if args.wandb_tracking:
        # Create PyTorch Lightning Trainer with Weights and Biases tracking
        run_name = f"{args.wandb_run_base_name}_{args.model_type}_bs-{args.batch_size}_lr-{args.lr: .5f}_dropout-{args.dropout_rate: .2f}_filts1-{args.num_filters_block_1}"
        wandb_logger = pl.loggers.WandbLogger(
            project="brain-mri-segmentation", log_model=True, name=run_name
        )
        hyperparameters = {
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "dropout_rate": args.dropout_rate,
            "num_filters_block_1": args.num_filters_block_1,
        }
        wandb_logger.experiment.config.update(hyperparameters)

        log_predictions_callback = BrainSegPredictionLogger()
        wandb_logger.watch(model, log_freq=1000)

        trainer = pl.Trainer(
            precision=args.precision,
            accelerator=accelerator,
            devices=1,
            max_epochs=args.max_epochs,
            default_root_dir=args.output_dir,
            logger=[wandb_logger],
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                swa_callback,
                log_predictions_callback,
                lr_callback,
            ],
            gradient_clip_val=0.5,
        )
    else:
        # Create PyTorch Lightning Trainer without Weights & Biases tracking
        pl_logger = pl.loggers.CSVLogger(save_dir="./pl-outputs")

        trainer = pl.Trainer(
            precision=args.precision,
            accelerator=accelerator,
            devices=1,
            max_epochs=args.max_epochs,
            default_root_dir=args.output_dir,
            logger=[pl_logger],
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                swa_callback,
                lr_callback,
            ],
            gradient_clip_val=0.5,
        )

    # Train the model
    logger.info("INFO: Starting model training...")
    trainer.fit(model, datamodule=brain_dm)
    logger.info("INFO: Finished model training...")

    if args.wandb_tracking:
        wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train PyTorch model on AWS Sagemaker for brain MRI segmentation."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="Path to training data (default: os.environ['SM_CHANNEL_TRAIN'] ",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=os.environ["SM_CHANNEL_VAL"],
        help="Path to validation data (default: os.environ['SM_CHANNEL_VAL'] ",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Directory to save model (default: os.environ['SM_MODEL_DIR'] ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
        help="Directory to save output data (default: os.environ['SM_OUTPUT_DATA_DIR'] ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="DataLoader batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Optimizer learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to run model training (default: 50)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout ratio.",
    )
    parser.add_argument(
        "--num_filters_block_1",
        type=int,
        default=32,
        help="Number of filters to use for first block of network (default: 32)",
    )
    parser.add_argument(
        "--fast_mode",
        type=str2bool,
        default=True,
        help="Whether to run fast training mode (e.g. moving all data to GPU).",
    )
    parser.add_argument(
        "--wandb_tracking",
        type=str2bool,
        default=False,
        help="Whether to track experiments with Weights & Biases (default: False)",
    )
    parser.add_argument(
        "--wandb_run_base_name",
        type=str,
        default="run",
        help="Base name to use for W&B run name (default: run)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for DataLoader (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=155,
        help="Value to use for seed.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        help="Type of model architecture. Either 'unet' or 'unet-attention' (default: unet)",
    )
    parser.add_argument(
        "--cache_rate",
        type=float,
        default=1.0,
        help="Proportion of data to load into memory. Default is 1 (all data).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Whether to run 16 or 32 bit precision training. Default is 32 bit.",
    )
    args = parser.parse_args()

    main(args)
