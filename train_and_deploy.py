import logging
import os

import yaml

from src.data import (
    create_kaggle_token_file,
    setup_brain_mri_dataset,
    upload_data_to_s3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("./config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    # PREPARE DATASET
    create_kaggle_token_file()
    setup_brain_mri_dataset(data_root_path=config["DATA_ROOT_PATH"])
    if config["RUN_MODE"] == "sagemaker":
        upload_data_to_s3(
            bucket_name=config["S3_BUCKET_NAME"],
            local_dataset_path=os.path.join(
                config["DATA_ROOT_PATH"], "brain-mri-dataset"
            ),
        )


if __name__ == "__main__":
    main()
