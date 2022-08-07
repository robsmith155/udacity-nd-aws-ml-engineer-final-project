import math
import os
import pathlib
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import EnsureChannelFirstd, EnsureTyped, LoadImaged
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
