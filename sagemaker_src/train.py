import argparse
import os

import numpy as np
import pytorch_lightning as pl
from brain_datamodule import BrainMRIData, monai_transformations
from brain_model import MyModel
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImage,
    LoadImaged,
    LoadImageD,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ToDeviced,
)
from torch.optim import AdamW


def main(args):
    # train_transforms = Compose([
    #     LoadImaged(keys=['image', 'label']),
    #     EnsureChannelFirstd(keys=["image", "label"]),
    #     #ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0, channel_wise=True),
    #     NormalizeIntensityd(keys=['label'], subtrahend=0, divisor=255.),
    #     NormalizeIntensityd(keys=["image"], subtrahend=np.array([23.78265792, 21.26016139, 22.85558883]), divisor=np.array([29.7563088, 28.0201569, 30.0995603]), channel_wise=True),
    #     EnsureTyped(keys=["image", "label"]),
    #     ToDeviced(keys=['image', 'label'], device='cuda'),
    #     RandFlipd(keys=['image', 'label'])
    # ])
    # train_transforms = Compose([
    #     LoadImaged(keys=['image', 'mask']),
    #     EnsureChannelFirstd(keys=['image', 'mask']),
    #     EnsureTyped(keys=["image", "mask"]),
    #     ToDeviced(keys=['image', 'mask'], device='cuda:0'),
    #     RandRotated(keys=['image', 'mask'], range_x=0.26, range_y=0.26, prob=0.5),
    #     ScaleIntensityRanged(keys=['image', 'mask'], a_min=0., a_max=255., b_min=0., b_max=1.),
    #     RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.5),
    #     RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
    #     RandGaussianNoised(keys=['image'], mean=0., std=0.1, prob=0.5)
    # ])

    # val_transforms = Compose([
    #     LoadImaged(keys=['image', 'label']),
    #     EnsureChannelFirstd(keys=["image", "label"]),
    #     #ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0, channel_wise=True),
    #     NormalizeIntensityd(keys=["image"], subtrahend=np.array([23.78265792, 21.26016139, 22.85558883]), divisor=np.array([29.7563088, 28.0201569, 30.0995603]), channel_wise=True),
    #     NormalizeIntensityd(keys=['label'], subtrahend=0, divisor=255.),
    #     EnsureTyped(keys=["image", "label"]),
    #     ToDeviced(keys=['image', 'label'], device='cuda')   # have device check at start and replace here with DEVICE
    # ])

    # val_transforms = Compose([
    #     LoadImaged(keys=['image', 'mask']),
    #     EnsureChannelFirstd(keys=['image', 'mask']),
    #     EnsureTyped(keys=["image", "mask"]),
    #     ScaleIntensityRanged(keys=['image', 'mask'], a_min=0., a_max=255., b_min=0., b_max=1.),
    #     ToDeviced(keys=['image', 'mask'], device='cuda:0')
    # ])

    train_transforms, val_transforms = monai_transformations(fast_mode=True)

    brain_dm = BrainMRIData(
        train_dir=args.train_data_dir,
        val_dir=args.val_data_dir,
        cache_rate=1.0,
        num_workers=8,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=args.batch_size,
        fast_mode=True,
    )

    unet = UNet(
        dimensions=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    max_epochs = 40
    loss_function = DiceFocalLoss(to_onehot_y=False, sigmoid=True)
    optimizer = AdamW

    model = MyModel(
        net=unet,
        criterion=loss_function,
        learning_rate=1e-4,
        batch_size=args.batch_size,
        optimizer_class=optimizer,
        # lr_scheduler=lr_scheduler
        # batch_size=16
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_mean_dice",
        mode="max",
        dirpath=args.model_dir,
        # filename="model-{epoch:02d}-{val_mean_dice:.2f}",
        filename="model-{epoch:02d}",
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="max",
    )

    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    # trainer = pl.Trainer(precision=16, accelerator='gpu', devices=1, max_epochs=40,
    #     default_root_dir='./checkpoints', callbacks=[early_stop_callback],
    #     logger=pl.loggers.CSVLogger(save_dir=f"{args.output_dir}/logs/"), gradient_clip_val=0.5)

    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.CSVLogger(save_dir=args.output_dir),
        gradient_clip_val=0.5,
    )

    # trainer = pl.Trainer(precision=16, accelerator='cpu', devices=1, max_epochs=40,
    #     default_root_dir='./checkpoints', callbacks=[checkpoint_callback, early_stop_callback],
    #     logger=pl.loggers.CSVLogger(save_dir="logs/"), gradient_clip_val=0.5)

    # trainer = pl.Trainer(precision=16, accelerator='gpu', devices=1, max_epochs=4,
    #     default_root_dir=args.output_dir, callbacks=[checkpoint_callback, early_stop_callback],
    #      gradient_clip_val=0.5)

    trainer.fit(model, datamodule=brain_dm)


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

    args = parser.parse_args()  # check what parse_known_args() does

    main(args)
