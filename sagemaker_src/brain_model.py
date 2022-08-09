import pytorch_lightning as pl
import torch
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose, EnsureType


class MyModel(pl.LightningModule):
    def __init__(
        self,
        net,
        criterion,
        learning_rate,
        batch_size,
        optimizer_class=None,
        lr_scheduler=None,
    ):
        super().__init__()
        self.net = net
        self.lr = learning_rate
        self.batch_size = torch.IntTensor(
            [batch_size]
        )  # Needed due to issues with Lightning logging?

        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean"
        )
        self.dice_metric_batch = DiceMetric(
            include_background=True, reduction="mean_batch"
        )

        self.save_hyperparameters(logger=False)
        self.example_input_array = torch.zeros(
            self.batch_size.item(), 3, 256, 256
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=self

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), 1e-4, weight_decay=1e-5)
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=1e-5
        )
        # lr_schedulers = {"scheduler": self.lr_scheduler(optimizer), "monitor": "metric_to_track"}
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10
            )
        }
        return [optimizer], [lr_scheduler]

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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y_true = self.infer_batch(batch)
        loss = self.criterion(y_hat, y_true)
        self.log("batch_size", self.batch_size)
        self.log("val_loss", loss, prog_bar=True)
        post_trans = Compose(
            [
                EnsureType(),
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5),
            ]
        )
        y_hat = post_trans(y_hat)
        # val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        self.dice_metric(y_pred=y_hat, y=y_true)
        self.dice_metric_batch(y_pred=y_hat, y=y_true)
        metric = self.dice_metric.aggregate().item()
        # metric_values.append(metric)
        # metric_batch = self.dice_metric_batch.aggregate() # CHECK THIS

        self.log("val_mean_dice", metric, prog_bar=True)  # CHECK THIS

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return loss

    def forward(self, x):
        return self.net(x)
