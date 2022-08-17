""" Create PyTorch Lightning datamodule

This module contains all the code needed to generate the PyTorch Lightning datamodule for the Brain MRI segmentation dataset.
"""
import logging
import os
import pathlib
from typing import List, Optional, Tuple, Union

import monai
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, ThreadDataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    ToDeviced,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def extract_input_filepaths(
    data_dirs: List[Union[str, pathlib.Path]]
) -> List[dict]:
    """Extracts the input image and segmentation mask filepaths as dictionaries from the specified folders.

    Args:
        data_dirs (List[Union[str, pathlib.Path]]): Path containing data directories.

    Returns:
        List[dict]t: List of dictionaries containing paths to input images and corrsponding masks
    """
    all_files = []
    for dir in data_dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith("_mask.tif") is False:
                    file_mask = file[:-4] + "_mask.tif"
                    if file_mask in files:
                        all_files.append(
                            {
                                "image": f"{root}/{file}",
                                "mask": f"{root}/{file_mask}",
                            }
                        )
    return all_files


def monai_transformations(
    fast_mode: bool = False, device: str = "cuda:0"
) -> Tuple[Compose, Compose]:
    """Creates data transformation pipelines for Monai.

    Args:
        fast_mode (bool, optional): Whether training is run in fast mode with all data moved to GPU. Defaults to False.
        device (str, optional): Device where training will be run. Defaults to "cuda:0".

    Returns:
        Tuple[Compose, Compose]: A tuple of Monai Compose transforms for training and validation data.
    """
    # TRAINING TRANSFORMS
    train_transforms = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys=["image", "mask"],
            a_min=0.0,
            a_max=255.0,
            b_min=0.0,
            b_max=1.0,
        ),
    ]

    if fast_mode:
        fast_transforms = [
            EnsureTyped(
                keys=["image", "mask"], data_type="tensor", track_meta=False
            ),
            ToDeviced(keys=["image", "mask"], device=device),
        ]

        [train_transforms.append(transform) for transform in fast_transforms]

    else:
        train_transforms.append(
            EnsureTyped(keys=["image", "mask"], data_type="tensor")
        )

    random_transforms = [
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
        RandRotated(
            keys=["image", "mask"], range_x=0.26, range_y=0.26, prob=0.5
        ),
        RandSpatialCropd(
            keys=["image", "mask"], roi_size=(224, 224), random_size=False
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.5),
    ]

    [train_transforms.append(transform) for transform in random_transforms]

    # VALIDATION TRANSFORMS
    val_transforms = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys=["image", "mask"],
            a_min=0.0,
            a_max=255.0,
            b_min=0.0,
            b_max=1.0,
        ),
        CenterSpatialCropd(keys=["image", "mask"], roi_size=(224, 224)),
    ]

    if fast_mode:
        fast_transforms = [
            EnsureTyped(
                keys=["image", "mask"], data_type="tensor", track_meta=False
            ),
            ToDeviced(keys=["image", "mask"], device=device),
        ]

        [val_transforms.append(transform) for transform in fast_transforms]
    else:
        val_transforms.append(
            EnsureTyped(keys=["image", "mask"], data_type="tensor")
        )

    return Compose(train_transforms), Compose(val_transforms)


class BrainMRIData(pl.LightningDataModule):
    """PyTorch Lightning datamodule that encapsulates all the data loading and proceessing steps."""

    def __init__(
        self,
        train_data_dir: Union[str, pathlib.Path],
        val_data_dir: Union[str, pathlib.Path],
        train_transforms: monai.transforms.Compose,
        val_transforms: monai.transforms.Compose,
        cache_rate: Optional[float] = 1.0,
        num_workers: Optional[int] = 8,
        batch_size: Optional[int] = 16,
        fast_mode: Optional[bool] = False,
    ):
        """
        Args:
            train_data_dir (Union[str, pathlib.Path]): Path containing training data.
            val_data_dir (Union[str, pathlib.Path]): Path containing validation data.
            train_transforms (monai.transforms.Compose): Monai transformation pipeline for training data.
            val_transforms (monai.transforms.Compose): Monai transformation pipeline for validation data.
            cache_rate (Optional[float], optional): Proportion of data to load into memory. Defaults to 1.0.
            num_workers (Optional[int], optional): Number of workers for dataloaders. Defaults to 8.
            batch_size (Optional[int], optional): Batch size for dataloaders. Defaults to 16.
            fast_mode (Optional[bool], optional): Whether to use fast mode, where all data moved to GPU. Defaults to False.
        """
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.batch_size = batch_size
        self.fast_mode = fast_mode

        if self.fast_mode is True:
            self.dataloader_workers = 0
        else:
            self.dataloader_workers = self.num_workers

    def setup(self, stage=None):
        """Create PyTorch Datasets using Monai."""
        train_folders = [
            f.path for f in os.scandir(self.train_data_dir) if f.is_dir()
        ]
        val_folders = [
            f.path for f in os.scandir(self.val_data_dir) if f.is_dir()
        ]

        self.train_files = extract_input_filepaths(train_folders)
        self.val_files = extract_input_filepaths(val_folders)

        self.train_ds = CacheDataset(
            data=self.train_files,
            transform=self.train_transform,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False,
        )

        self.val_ds = CacheDataset(
            data=self.val_files,
            transform=self.val_transform,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            copy_cache=False,
        )

    def train_dataloader(self):
        """Create Monai DataLoader for training data."""

        if self.fast_mode:
            return ThreadDataLoader(
                dataset=self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
        else:
            return DataLoader(
                dataset=self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.dataloader_workers,
                drop_last=True,
                pin_memory=True,
            )

    def val_dataloader(self):
        """Creates Monai DataLoader for validation data."""
        if self.fast_mode:
            return ThreadDataLoader(
                dataset=self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        else:
            return DataLoader(
                dataset=self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.dataloader_workers,
                drop_last=False,
                pin_memory=True,
            )


class BrainMRIDataOptuna(BrainMRIData):
    """PyTorch Lightning datamodule that encapsulates proceessing steps.

    Used for Optuna HPO search so that data isn't repeatedly loaded into memory.
    """

    def __init__(
        self,
        train_data_dir: Union[str, pathlib.Path],
        val_data_dir: Union[str, pathlib.Path],
        train_transforms: Compose,
        val_transforms: Compose,
        cache_rate: Optional[float],
        num_workers: Optional[int],
        batch_size: Optional[int],
        fast_mode: Optional[bool],
        train_files: List[dict],
        val_files: List[dict],
        train_ds: monai.data.CacheDataset,
        val_ds: monai.data.CacheDataset,
    ):

        super().__init__(
            train_data_dir,
            val_data_dir,
            train_transforms,
            val_transforms,
            cache_rate,
            num_workers,
            batch_size,
            fast_mode,
        )

        self.train_files = train_files
        self.val_files = val_files
        self.train_ds = train_ds
        self.val_ds = val_ds

    def setup(self, stage=None):
        pass


def create_data_module(
    train_data_dir: Union[str, pathlib.Path],
    val_data_dir: Union[str, pathlib.Path],
    batch_size: Optional[int] = 16,
    cache_rate: Optional[float] = 1.0,
    num_workers: Optional[int] = 8,
    fast_mode: Optional[bool] = False,
) -> pl.LightningDataModule:
    """Create Monai transformations and combines with inputs to generate PyTorch Lightning datamodule.

    Args:
        train_data_dir (Union[str, pathlib.Path]):  Path containing training data.
        val_data_dir (Union[str, pathlib.Path]):  Path containing validation data.
        batch_size (Optional[int], optional): batch size for dataloaders. Defaults to 16.
        cache_rate (Optional[float], optional): Proportion of data to load into memory. Defaults to 1.0.
        num_workers (Optional[int], optional): Number of workers for dataloaders. Defaults to 8.
        fast_mode (Optional[bool], optional): Whether to use fast mode, where all data moved to GPU. Defaults to False.

    Returns:
        pl.LightningDataModule: PyTorch Lightning datamodule.
    """
    logger.info("INFO: Setting up PyTorch Data Module.")
    train_transforms, val_transforms = monai_transformations(
        fast_mode=fast_mode
    )

    brain_dm = BrainMRIData(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        cache_rate=cache_rate,
        num_workers=num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=batch_size,
        fast_mode=fast_mode,
    )
    logger.info("INFO: Finished setting up BrainMRIData Module.")
    return brain_dm
