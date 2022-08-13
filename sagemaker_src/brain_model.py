import logging

import numpy as np
import pytorch_lightning as pl
import torch
from monai.metrics import DiceMetric
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
    def __init__(self, net, criterion, learning_rate, batch_size):
        super().__init__()
        self.net = net
        self.lr = learning_rate
        self.batch_size = torch.FloatTensor(
            [batch_size]
        )  # Needed due to issues with Lightning logging?
        self.criterion = criterion
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean"
        )
        self.save_hyperparameters(ignore=["net", "criterion"])
        self.example_input_array = torch.zeros(batch_size, 3, 256, 256)
        self.val_dice_all = []

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
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        output_pred_table = False

        if pl_module.current_epoch == 0:
            output_pred_table = True
        else:
            if pl_module.val_dice_all[-1] > max(pl_module.val_dice_all[:-1]):
                output_pred_table = True

        # Only output predictions if validation score has improved
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

            imgs, y_true, y_pred = [], [], []
            for sample in val_samples:
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
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), wandb.Image(y_i), wandb.Image(y_pred)]
                for x_i, y_i, y_pred in list(zip(imgs, y_true, y_pred))
            ]
            table_name = f"Epoch {pl_module.current_epoch} predictions (Mean val Dice score: {pl_module.val_dice_all[-1]: .3f})"
            trainer.logger.experiment.log(
                {table_name: wandb.Table(data=data, columns=columns)},
                commit=False,
            )
