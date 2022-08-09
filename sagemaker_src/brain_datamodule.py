import os

import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, ThreadDataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ToDeviced,
)


def extract_input_filepaths(data_dirs):
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


def monai_transformations(fast_mode=False, device="cuda:0"):
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
        RandRotated(
            keys=["image", "mask"], range_x=0.26, range_y=0.26, prob=0.5
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
    def __init__(
        self,
        train_dir,
        val_dir,
        cache_rate,
        num_workers,
        train_transforms,
        val_transforms,
        batch_size,
        fast_mode=False,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        # self.test_dir = test_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.batch_size = batch_size
        self.fast_mode = fast_mode

        if self.fast_mode:
            self.dataloader_workers = 0
        else:
            self.dataloader_workers = self.num_workers

    def setup(self, stage=None):
        train_folders = [
            f.path for f in os.scandir(self.train_dir) if f.is_dir()
        ]
        val_folders = [f.path for f in os.scandir(self.val_dir) if f.is_dir()]
        # test_folders = [f.path for f in os.scandir(self.test_dir) if f.is_dir()]

        self.train_files = extract_input_filepaths(train_folders)
        self.val_files = extract_input_filepaths(val_folders)
        # self.test_files = extract_input_filepaths(test_folders)

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
        # Make num_workers 0 if using GPU and data already moved there. CHANGE BELOW TO IF STATEMENT.
        # return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        if self.fast_mode:
            return ThreadDataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.dataloader_workers,
                drop_last=True,
            )

    def val_dataloader(self):
        if self.fast_mode:
            return ThreadDataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        else:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.dataloader_workers,
                drop_last=False,
            )
