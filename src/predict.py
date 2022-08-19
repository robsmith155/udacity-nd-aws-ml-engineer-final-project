from typing import List, Tuple

import numpy as np
import torch
from monai.transforms import (
    AsChannelFirst,
    CenterSpatialCrop,
    Compose,
    EnsureType,
    ScaleIntensityRange,
)
from PIL import Image
from torchmetrics.functional import dice_score
from tqdm.auto import tqdm


def predict_segmentation_masks(file_list: List[dict], predictor):
    """Predicts segmentation masks for list of input files using endpoint.

    Submits the input images to the endpoint to get teh predicted segmentation masks. Also
    outputs the corresponding input MRI image and true mask as Numpy arrays. The Dice score
    is also provided.

    Args:
        file_list (List[dict]): List of dictionaries containing the path to the input MRI image ad corresponding segmentation mask.
        predictor (sagemaker.PyTorchModel.predict): SageMaker deployed endpoint for inference.

    Returns:
        _type_: _description_
    """
    dice_score_all = []
    img_all = np.empty(shape=(len(file_list), 224, 224, 3))
    mask_all = np.empty(shape=(len(file_list), 224, 224, 1))
    preds_all = np.empty(shape=(len(file_list), 224, 224, 1))

    for i, file in enumerate(tqdm(file_list)):

        img = np.array(Image.open(file["image"]))
        mask = np.array(Image.open(file["mask"]))[:, :, None]

        # Send image data to endpoint to make prediction
        pred = predictor.predict(img)
        pred_torch = torch.tensor(pred)[None, :, :]
        pred_torch = torch.permute(pred_torch, (0, 2, 1))

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

        img = img_transforms(img)
        mask = mask_transforms(mask)
        mask = mask.as_tensor()
        mask = mask.to(torch.int32)

        dice_score_all.append(dice_score(pred_torch, mask, bg=True).item())

        img = img.detach().cpu().numpy()
        img = np.transpose(img, (2, 1, 0))

        mask = mask.detach().cpu().numpy()
        mask = np.transpose(mask, (2, 1, 0))

        img_all[i] = img
        mask_all[i] = mask
        preds_all[i] = pred[:, :, None]

    return dice_score_all, img_all, mask_all, preds_all
