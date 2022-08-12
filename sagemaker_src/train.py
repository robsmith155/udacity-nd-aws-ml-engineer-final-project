import argparse
import os

import pytorch_lightning as pl
import torch
from brain_datamodule import BrainMRIData, monai_transformations
from brain_model import MyModel
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.networks.nets import UNet


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

    train_transforms, val_transforms = monai_transformations(fast_mode=True)

    brain_dm = BrainMRIData(
        train_dir=args.train_data_dir,
        val_dir=args.val_data_dir,
        cache_rate=1.0,
        num_workers=8,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=args.batch_size,
        fast_mode=args.fast_mode,
    )
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
    optimizer = torch.optim.AdamW

    model = MyModel(
        net=unet,
        criterion=loss_function,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        optimizer_class=optimizer,
        # lr_scheduler=lr_scheduler
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_mean_dice",
        mode="max",
        dirpath=args.model_dir,
        filename="model",
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="max",
    )

    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.CSVLogger(save_dir=args.output_dir),
        gradient_clip_val=0.5,
    )

    trainer.fit(model, datamodule=brain_dm)


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.ckpt")

    model = MyModel.load_from_checkpoint(checkpoint_path=model_path)
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
    args = parser.parse_args()  # check what parse_known_args() does

    main(args)
