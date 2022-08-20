import logging
import sys
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import AttentionUnet, UNet
from monai.transforms import (
    Activations,
    AsChannelFirst,
    AsDiscrete,
    CenterSpatialCrop,
    Compose,
    EnsureType,
    ScaleIntensityRange,
)
from PIL import Image

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class BrainMRIModel(pl.LightningModule):
    """PyTorch Lightning module to organize code for training model."""

    def __init__(
        self,
        model_type: Optional[str] = "unet",
        learning_rate: Optional[float] = 0.001,
        batch_size: Optional[int] = 16,
        dropout_rate: Optional[float] = 0.1,
        num_filters_block_1: Optional[int] = 16,
        wandb_tracking: Optional[bool] = False,
    ):
        """_summary_

        Args:
            model_type (Optional[str], optional): Model architecture to use. Either 'unet' or 'unet-attention'. Defaults to 'unet'.
            learning_rate (Optional[float], optional): Optimizer learning rate. Defaults to 0.001.
            batch_size (Optional[int], optional): dataloader batch size. Defaults to 16.
            dropout_rate (Optional[float], optional): Dropout rate. Defaults to 0.1.
            num_filters_block_1 (Optional[int], optional): Number of filters to use in first block of model. Defaults to 16.
            wandb_tracking (Optional[bool]): Whether tracking with Weights and Biases. Default is False.
        """
        super().__init__()
        self.model_type = model_type
        self.lr = learning_rate
        self.batch_size = torch.FloatTensor(
            [batch_size]
        )  # Needed due to issues with Lightning logging
        self.dropout_rate = dropout_rate
        self.filters1 = num_filters_block_1
        self.wandb_tracking = wandb_tracking
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean"
        )
        self.net = self.create_net()
        self.example_input_array = torch.zeros(batch_size, 3, 256, 256)
        self.val_dice_all = []

        self.save_hyperparameters(
            ignore=["net", "criterion"]
        )  # Training very slow if these ar enot ignored

    def create_net(self):
        """Create instance of network to be trained."""
        channels = (
            self.filters1,
            self.filters1 * 2,
            self.filters1 * 4,
            self.filters1 * 8,
            self.filters1 * 16,
        )

        if self.model_type == "unet":
            network = UNet(
                dimensions=2,
                in_channels=3,
                out_channels=1,
                channels=channels,
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=self.dropout_rate,
            )
        elif self.model_type == "unet-attention":
            network = AttentionUnet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=channels,
                strides=(2, 2, 2, 2),
                dropout=self.dropout_rate,
            )
        else:
            logger.error(
                "ERROR: You entered an invalid entry for model_type. Please enter either 'unet' or 'unet-attention'"
            )
            sys.exit()
        return network

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-5
        )

        return optimizer

    def prepare_batch(self, batch):
        images = batch["image"]
        labels = batch["mask"]
        return images, labels

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("batch_size", self.batch_size)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if (self.current_epoch == 0) and (self.wandb_tracking is True):
            wandb.define_metric("val_mean_dice", summary="max")
        val_batch_size = len(batch)
        images = batch["image"]
        y_hat, y_true = self.infer_batch(batch)
        loss = self.criterion(y_hat, y_true)
        self.log(
            "batch_size", torch.tensor(val_batch_size, dtype=torch.float32)
        )
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        post_trans = Compose(
            [
                EnsureType(),
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5),
            ]
        )
        y_hat = post_trans(y_hat)
        self.dice_metric(y_pred=y_hat, y=y_true)

        return {
            "val_loss": loss,
            "val_samples": val_batch_size,
            "val_images": images,
            "val_masks": y_true,
            "val_preds": y_hat,
        }

    def validation_epoch_end(self, outputs):

        val_mean_dice = self.dice_metric.aggregate().item()

        self.log("val_mean_dice", val_mean_dice, prog_bar=True)
        logger.info(f"Epoch: {self.current_epoch}; Val_dice={val_mean_dice};")
        self.val_dice_all.append(val_mean_dice)
        self.dice_metric.reset()

        return {"val_mean_dice": val_mean_dice}

    def forward(self, x):
        return self.net(x)


class BrainSegPredictionLogger(pl.Callback):
    """Callback to log predictions in Weights & Biases.

    This is a custom callback that will be triggered at the end of each validation epoch. If the validation score improved, predictions from
    a sample of the validation data will be generated and logged as a Table in Weights & Biases.

    Note: This method could be improved by passing the predicted data from the validation_epoch_end step. That way I don't need to load and run
    the data again. However, I haven't figured out how to do that yet.

    """

    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        # Check if validation score has improved
        output_pred_table = False

        if pl_module.current_epoch == 0:
            output_pred_table = True
        else:
            if pl_module.val_dice_all[-1] > max(pl_module.val_dice_all[:-1]):
                output_pred_table = True

        # Output predictions if validation score has improved
        if output_pred_table:
            val_files = trainer.datamodule.val_files
            accessed_mapping = map(
                val_files.__getitem__, list(np.arange(10, len(val_files), 20))
            )
            val_samples = list(accessed_mapping)

            img_transforms = Compose(
                [
                    AsChannelFirst(),
                    ScaleIntensityRange(
                        a_min=0.0,
                        a_max=255.0,
                        b_min=0.0,
                        b_max=1.0,
                    ),
                    CenterSpatialCrop(roi_size=(224, 224)),
                    EnsureType(data_type="tensor"),
                ]
            )

            mask_transforms = Compose(
                [
                    AsChannelFirst(),
                    CenterSpatialCrop(roi_size=(224, 224)),
                    EnsureType(data_type="tensor"),
                ]
            )

            names, imgs, y_true, y_pred = [], [], [], []
            for sample in val_samples:
                slice_name = sample["image"].split("/")[-1]
                names.append(slice_name)
                img = np.array(Image.open(sample["image"]))
                mask = np.array(Image.open(sample["mask"]))[:, :, None]

                img = img_transforms(img)
                mask = mask_transforms(mask)

                img_model = img.as_tensor()[None, :, :, :]
                result = pl_module(img_model.to("cuda:0")).squeeze()

                pred = result.detach().cpu().numpy()
                pred = np.where(pred < 0, 0, 1)
                pred = np.transpose(pred, (1, 0))
                y_pred.append(pred)

                img = img.detach().cpu().numpy()
                img = np.transpose(img, (2, 1, 0))
                imgs.append(img)

                mask = mask.as_tensor()
                mask = mask.detach().cpu().numpy()
                mask = np.transpose(mask, (2, 1, 0))
                y_true.append(mask)

            # Log as W&B Table
            columns = ["slice name", "image", "ground truth", "prediction"]
            data = [
                [name, wandb.Image(x_i), wandb.Image(y_i), wandb.Image(y_pred)]
                for name, x_i, y_i, y_pred in list(
                    zip(names, imgs, y_true, y_pred)
                )
            ]
            table_name = f"Epoch {pl_module.current_epoch} predictions (Mean val Dice score: {pl_module.val_dice_all[-1]: .3f})"
            trainer.logger.experiment.log(
                {table_name: wandb.Table(data=data, columns=columns)},
                commit=False,
            )

            del (
                imgs,
                y_true,
                y_pred,
                mask,
                pred,
                img,
                data,
                val_files,
                val_samples,
            )
