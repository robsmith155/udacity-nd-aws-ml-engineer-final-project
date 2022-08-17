import logging
import os
import sys

import numpy as np
import torch
from brain_model import BrainMRIModel
from monai.transforms import (
    AsChannelFirst,
    CenterSpatialCrop,
    Compose,
    EnsureType,
    ScaleIntensityRange,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):

    model_path = os.path.join(model_dir, "model.ckpt")

    logger.info(f"INFO: Loading model checkpoint from {model_path}")
    model = BrainMRIModel.load_from_checkpoint(model_path)
    logger.info("INFO: Model loaded successfully...")
    return model


## INPUT_FN
## Ideally I would serialize the data as bytes and send to make a request, but function below currently doesnt work
# def input_fn(request_body, content_type='image/tiff'):
#     logger.info('Deserializing input data')
#     logger.info(f'Request content type: {type(request_body)}')
#     if content_type == 'image/tiff':
#         logger.info('Loading image')
#         return Image.open(io.BytesIO(request_body))
#     elif content_type == 'application/tiff':
#         img_request = requests.get(request_body['url'], stream=True)
#         return Image.open(io.BytesIO(img_request.content))
#     raise Exception(f'Unsupported content type ({type(request_body)}). Expected image/tiff')


def predict_fn(input_object, model):
    logger.info("Starting predict function")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

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

    logger.info("Passing data through image transformations")
    img = img_transforms(input_object)

    with torch.no_grad():
        logger.info(f"Making prediction on input object...")
        result = model(img.as_tensor()[None, :, :, :].to(device)).squeeze()
        result = result.detach().cpu().numpy()
        pred = np.where(result < 0, 0, 1)
        pred = np.transpose(pred, (1, 0))
        logger.info(
            f"Prediction made from model with output dimensions of {pred.shape}"
        )

    return pred
