import argparse
import logging
import os

import pytorch_lightning as pl
from brain_datamodule import BrainMRIData, monai_transformations
from brain_model import BrainMRIModel, BrainSegPredictionLogger
from monai.losses import DiceFocalLoss
from monai.networks.nets import UNet

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Function below copied from : https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):

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

    train_transforms, val_transforms = monai_transformations(
        fast_mode=args.fast_mode
    )
    logger.info("INFO: Created MONAI transformations.")

    brain_dm = BrainMRIData(
        train_dir=args.train_data_dir,
        val_dir=args.val_data_dir,
        cache_rate=1.0,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=args.batch_size,
        fast_mode=args.fast_mode,
    )
    logger.info("INFO: Created BrainMRIData datamodule.")

    filters1 = args.num_filters_block_1

    unet = UNet(
        dimensions=2,
        in_channels=3,
        out_channels=1,
        channels=(
            filters1,
            filters1 * 2,
            filters1 * 4,
            filters1 * 8,
            filters1 * 16,
        ),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=args.dropout_rate,
    )

    loss_function = DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    model = BrainMRIModel(
        net=unet,
        criterion=loss_function,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    logger.info("INFO: Created BrainMRIModel instance.")

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

    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    if args.wandb_tracking:
        run_name = f"{args.wandb_run_base_name}_bs-{args.batch_size}_lr-{args.lr: .5f}_dropout-{args.dropout_rate: .2f}_filts1-{args.num_filters_block_1}"
        wandb_logger = pl.loggers.WandbLogger(
            project="brain-mri-segmentation", log_model=True, name=run_name
        )
        log_predictions_callback = BrainSegPredictionLogger()
        wandb_logger.watch(model)

        trainer = pl.Trainer(
            precision=16,
            accelerator="gpu",
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
        pl_logger = pl.loggers.CSVLogger(save_dir="./lightning")

        trainer = pl.Trainer(
            precision=16,
            accelerator="gpu",
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

    logger.info("INFO: Starting model training...")
    trainer.fit(model, datamodule=brain_dm)
    logger.info("INFO: Finished model training...")

    if args.wandb_tracking:
        wandb.finish()


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.ckpt")

    model = BrainMRIModel.load_from_checkpoint(checkpoint_path=model_path)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch hyperparameter tuning"
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
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to run model training.",
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
        help="Number of filters to use first block of network.",
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
        help="Whether to track experiments with Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_run_base_name",
        type=str,
        default="run",
        help="Base name to use for W&B run name.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for DataLoader.",
    )
    args = parser.parse_args()  # check what parse_known_args() does

    main(args)
