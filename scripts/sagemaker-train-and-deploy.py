# THIS SCRIPT NEEDS TO BE COMPLETED
import logging
import os

import typer
import yaml
from sagemaker.pytorch.estimator import PyTorch

from src.data import (
    create_kaggle_token_file,
    setup_brain_mri_dataset,
    upload_data_to_s3,
)
from src.utils import set_sagemaker_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("./config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main(
    run_mode: str,
    # instance_type: str = None, #'ml.g4dn.xlarge'
):
    # PREPARE DATASET
    create_kaggle_token_file()
    setup_brain_mri_dataset(data_root_path=config["DATA_ROOT_PATH"])

    # UPLOAD DATA TO S3 BUCKET
    if run_mode == "sagemaker":
        upload_data_to_s3(
            bucket_name=config["S3_BUCKET_NAME"],
            local_dataset_path=os.path.join(
                config["DATA_ROOT_PATH"], "brain-mri-dataset"
            ),
        )

    # TRAIN MODEL
    logger.info("Setting up model training...")
    sagemaker_settings = set_sagemaker_settings(run_mode=run_mode)
    logger.info(f"Sagemaker settings: {sagemaker_settings}")
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="./sagemaker_src",
        role=sagemaker_settings["role_arn"],
        py_version="py38",
        framework_version="1.11.0",
        instance_type=sagemaker_settings["instance_type"],
        instance_count=1,
        use_spot_instances=sagemaker_settings["use_spot_instances"],
        max_run=sagemaker_settings["max_run"],
        max_wait=sagemaker_settings["max_wait"],
        sagemaker_session=sagemaker_settings["sagemaker_session"],
        output_path=sagemaker_settings["output_path"],
        checkpoint_s3_uri=sagemaker_settings["checkpoint_s3_uri"],
        base_job_name=sagemaker_settings["base_job_name"],
    )

    logger.info("Starting model training ...")
    estimator.fit(
        {
            "train": sagemaker_settings["training_input_path"],
            "val": sagemaker_settings["validation_input_path"],
        },
        wait=True,
    )
    logger.info("Model training finished.")

    predictor = estimator.deploy(
        initial_instance_count=1, instance_type="local", wait=False
    )


if __name__ == "__main__":
    typer.run(main)
