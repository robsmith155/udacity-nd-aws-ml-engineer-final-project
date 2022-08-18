import math
import os
import pathlib
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import monai
import numpy as np
import sagemaker
import torch
from monai.transforms import (
    AsChannelFirst,
    CenterSpatialCrop,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRange,
)
from monai.visualize import blend_images
from PIL import Image

from src.data import extract_patient_filepaths


def plot_patient_mri(patient_data_path: Union[str, pathlib.Path]) -> None:
    """Plots all MRI slices from patient data folder.

    Args:
        patient_data_path (Union[str, pathlib.Path]): Path containing patient MRI image data.
    """
    image_paths = extract_patient_filepaths(patient_data_path)
    num_rows = math.ceil(len(image_paths) / 6)
    patient_folder = os.path.basename(patient_data_path)

    plt.figure(figsize=(24, 4 * num_rows))
    plt.suptitle(f"Patient: {patient_folder}")
    for i, image in enumerate(image_paths):
        im = Image.open(image["image"])
        im = np.array(im)

        plt.subplot(num_rows, 6, i + 1)
        plt.title(f"Slice {i}")
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.imshow(im)


def plot_patient_mri_mask_overlay_monai(
    patient_data_path: Union[str, pathlib.Path]
) -> None:
    """Plots all MRI slices from patient data folder with segmentation mask overlay.

    Args:
        patient_data_path (Union[str, pathlib.Path]): Path containing patient MRI image data.
    """
    image_paths = extract_patient_filepaths(patient_data_path)
    num_rows = math.ceil(len(image_paths) / 6)
    patient_folder = os.path.basename(patient_data_path)

    plt.figure(figsize=(24, 4 * num_rows))
    plt.suptitle(f"Patient: {patient_folder}")
    for i, data in enumerate(image_paths):
        data = LoadImaged(keys=["image", "mask"])(data)
        data = EnsureChannelFirstd(keys=["image", "mask"])(data)
        data = EnsureTyped(keys=["image", "mask"])(data)
        blend = blend_images(
            image=data["image"],
            label=data["mask"],
            alpha=0.5,
            cmap="hsv",
            rescale_arrays=True,
        )
        plt.subplot(num_rows, 6, i + 1)
        plt.title(f"Slice {i}")
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.imshow(torch.permute(blend[:, :, :], (2, 1, 0)))


def plot_patient_mri_mask_overlay(
    patient_data_path: Union[str, pathlib.Path]
) -> None:
    """Plots all MRI slices from patient data folder with segmentation mask overlay using MONAI blend.

    Args:
        patient_data_path (Union[str, pathlib.Path]): Path containing patient MRI image data.
    """
    image_paths = extract_patient_filepaths(patient_data_path)
    num_rows = math.ceil(len(image_paths) / 6)
    patient_folder = os.path.basename(patient_data_path)

    plt.figure(figsize=(24, 4 * num_rows))
    plt.suptitle(f"Patient: {patient_folder}")
    for i, data in enumerate(image_paths):
        im = Image.open(data["image"])
        im = np.array(im)
        mask = Image.open(data["mask"])
        mask = np.array(mask)
        mask = np.ma.masked_where(mask < 255, mask)
        mask = mask.squeeze()
        plt.subplot(num_rows, 6, i + 1)
        plt.title(f"Slice {i}")
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.imshow(
            im[:, :, 1], cmap="gray"
        )  # Told the FLAIR image is the second channel in README
        plt.imshow(mask, cmap="jet", vmin=0, vmax=1, alpha=0.35)


def plot_patient_slice_channels(
    patient_data_path: Union[str, pathlib.Path], slice_idxs: List[int]
) -> None:
    """Plots three MRI channels plus mask overlay for the specified slice numbers of a patient.

    Args:
        patient_data_path (Union[str, pathlib.Path]): Path containing patient MRI image data.
        slice_idxs (List[int]): Slice indexes to be plotted from patient.
    """
    img_paths = extract_patient_filepaths(patient_data_path)
    col_titles = ["Pre-contrast", "FLAIR", "Post-contrast", "FLAIR + mask"]
    row_titles = []
    for idx in slice_idxs:
        row_titles.append(f"Slice {idx}")

    fig, axes = plt.subplots(
        nrows=len(slice_idxs), ncols=4, figsize=(len(slice_idxs) * 4, 16)
    )

    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row, rotation=90, size="large")

    i = 1
    for idx in slice_idxs:
        im = Image.open(img_paths[idx]["image"])
        im = np.array(im)
        mask = Image.open(img_paths[idx]["mask"])
        mask = np.array(mask)
        mask = np.ma.masked_where(mask < 1, mask)
        for j in range(3):
            plt.subplot(len(slice_idxs), 4, i)
            plt.imshow(im[:, :, j], cmap="gray")
            i += 1

        plt.subplot(len(slice_idxs), 4, i)
        plt.imshow(im[:, :, 1], cmap="gray")
        plt.imshow(mask, alpha=0.3, vmin=0.1, vmax=1, cmap="jet")
        i += 1


# Function below from https://note.nkmk.me/en/python-pillow-concat-images/
def _get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def _combine_mri_mask(data):
    im1 = Image.open(data["image"])
    im2 = Image.open(data["mask"])
    im_combined = _get_concat_h(im1, im2)
    return im_combined


# Function below based on https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
def make_patient_data_gif(
    patient_dir_path: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path] = ".",
) -> None:
    """Create GIF of patient MRI or segmentation mask data.

    Args:
        patient_dir_path (Union[str, pathlib.Path]): Path containing patient MRI image data.

        save_dir (Union[str, pathlib.Path], optional): Directory to save GIF file to. Defaults to '.'.
    """
    patient_id = os.path.basename(patient_dir_path)
    patient_file_list = extract_patient_filepaths(patient_dir_path)
    frames = [_combine_mri_mask(data) for data in patient_file_list]
    frame_one = frames[0]
    output_filename = f"{patient_id}_slices.gif"
    output_path = os.path.join(save_dir, output_filename)
    frame_one.save(
        output_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        fps=10,
        loop=0,
    )
    print(f"GIF file output to {output_path}.")


def plot_monai_transformed_data(
    data_dict: dict,
    transpose: bool = True,
    convert_uint8: bool = True,
    raw_data: bool = False,
) -> None:
    """Plot of MRI image and segmentation mask output from MONAI transform

    Args:
        data_dict (dict): Dictionary containing the path to the image and mask data. Keys must be 'image' and 'mask'.
        transpose (bool, optional): Whether the data should be transposed (i.e. rotated). Defaults to True.
        convert_uint8 (bool, optional): Whether to convert the data to np.uint8. Defaults to True.
    """
    img = data_dict["image"].numpy()
    mask = data_dict["mask"].numpy()

    if convert_uint8:
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

    if transpose:
        img = np.transpose(img, (2, 1, 0))
        mask = np.transpose(mask, (2, 1, 0))

    if raw_data:
        img = np.transpose(img, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.colorbar()
    plt.title("MRI")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.colorbar()
    plt.title("Mask")


def plot_monai_pipeline_data(
    data_dict: dict, transform_pipeline: monai.transforms.Compose
) -> None:
    """Plots input image and mask and four different augmented views after data passed through MONAI transformation pipeline.

    Args:
        data_dict (dict): Dictionary containing the path to the image and mask data. Keys must be 'image' and 'mask'.
        transform_pipeline (monai.transforms.Compose): MONAI transformation pipeline.
    """
    raw_im = Image.open(data_dict["image"])
    raw_im = np.array(raw_im)
    raw_mask = Image.open(data_dict["mask"])
    raw_mask = np.array(raw_mask)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
    col_names = ["Original", "Aug1", "Aug2", "Aug3", "Aug4"]

    for ax, col in zip(axes[0], col_names):
        ax.set_title(col)

    axes[0, 0].imshow(raw_im)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(raw_mask)
    axes[1, 0].axis("off")

    for i in range(4):
        data_trans = transform_pipeline(data_dict)
        img = data_trans["image"].numpy()
        mask = data_trans["mask"].numpy()
        img = np.transpose(img, (2, 1, 0))
        mask = np.transpose(mask, (2, 1, 0))
        axes[0, i + 1].imshow(img)
        axes[0, i + 1].axis("off")
        axes[1, i + 1].imshow(mask)
        axes[1, i + 1].axis("off")

    fig.tight_layout()


def plot_endpoint_prediction(
    predictor: sagemaker.pytorch.model.PyTorchPredictor,
    image_path: Union[str, pathlib.Path],
    mask_path: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """Predict segmentation mask and plot results.

    This function will submit the input image (in image_path) to the specified SageMaker endpoint provide dby the predictor.

    It will then output the input MRI image with the predicted segmentation mask. If the true mask is available, this will also be plotted for comparison.

    Args:
        predictor (sagemaker.pytorch.model.PyTorchPredictor): Contains SageMaker endpoint to make prediction.
        image_path (Union[str, pathlib.Path]): Path to input MRI image you want to predict the segmentation mask for.
        mask_path (Optional[Union[str, pathlib.Path]], optional): Path to ground truth segmentation mask. Defaults to None.
    """
    # Load the image and mask data (if available) as a Numpy array
    img = np.array(Image.open(image_path))

    if mask_path is not None:
        mask = np.array(Image.open(mask_path))[:, :, None]

    # Send image data to endpoint to make prediction
    pred = predictor.predict(img)

    # Transform image and mask data
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

    img = img_transforms(img)
    if mask_path is not None:
        mask = mask_transforms(mask)

    img = img.detach().cpu().numpy()
    img = np.transpose(img, (2, 1, 0))

    if mask_path is not None:
        mask = mask.detach().cpu().numpy()
        mask = np.transpose(mask, (2, 1, 0))

    # Plot results
    if mask_path is not None:
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("MRI input")
        plt.imshow(img)

        plt.subplot(1, 3, 2)
        plt.title("True mask")
        plt.imshow(mask)

        plt.subplot(1, 3, 3)
        plt.title("Predicted mask")
        plt.imshow(pred)

    if mask_path is None:
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.title("MRI input")
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title("Predicted mask")
        plt.imshow(pred)
