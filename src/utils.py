import logging
import os
import pathlib
import sys
from typing import Optional, Union

import boto3
import git
import pandas as pd
import sagemaker
import yaml

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Resolve repo root path and add to path
git_repo = git.Repo(".", search_parent_directories=True)
PROJECT_ROOT_PATH = git_repo.working_dir

with open(f"{PROJECT_ROOT_PATH}/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PROJECT_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, config["DATA_DIR"])


def set_sagemaker_settings(
    run_mode,
    instance_type="ml.g4dn.xlarge",
    use_spot_instances=True,
    base_job_name="brain-mri-segmentation",
):
    if run_mode in ["local-all", "local-all-gpu"]:
        # Sagemaker settings for running training locally
        sagemaker_session = sagemaker.LocalSession()
        sagemaker_session.config = {
            "local": {"local_code": True}
        }  # Ensure full code locality, see: https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode
        role_arn = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"  # Dummy ARN
        training_input_path = (
            f"file://{PROJECT_DATA_PATH}/brain-mri-dataset/train"
        )
        validation_input_path = (
            f"file://{PROJECT_DATA_PATH}/brain-mri-dataset/val"
        )
        output_path = f"file://{PROJECT_ROOT_PATH}/sagemaker_outputs/"
        use_spot_instances = False
        max_run = None
        max_wait = None
        checkpoint_s3_uri = None
        base_job_name = base_job_name
        if run_mode == "local-all-gpu":
            instance_type = "local_gpu"
        else:
            instance_type = "local"
    elif run_mode == "local-submit":
        # Sagemaker settings for submitting to AWS job from local machine
        iam_client = boto3.client("iam")
        role_name = input(
            "Please enter the name of the Amazon SageMaker execution role that you want to use: "
        )
        role_arn = iam_client.get_role(RoleName=role_name)["Role"]["Arn"]
        training_input_path = f"s3://{config['S3_BUCKET_NAME']}/data/train"
        validation_input_path = f"s3://{config['S3_BUCKET_NAME']}/data/val/"
        output_path = f"s3://{config['S3_BUCKET_NAME']}/outputs"
        instance_type = instance_type
        use_spot_instances = use_spot_instances
        max_run = 3600
        if use_spot_instances:
            max_wait = 7200
        else:
            max_wait = None
        checkpoint_s3_uri = f"s3://{config['S3_BUCKET_NAME']}/checkpoints"
        base_job_name = base_job_name
        sagemaker_session = sagemaker.Session(
            default_bucket=config["S3_BUCKET_NAME"]
        )
    elif run_mode == "sagemaker":
        # Sagemaker settings if submitting job from SageMaker notebook or Studio
        sagemaker_session = sagemaker.Session()
        role_arn = sagemaker.get_execution_role()
        training_input_path = f"s3://{config['S3_BUCKET_NAME']}/data/train"
        validation_input_path = f"s3://{config['S3_BUCKET_NAME']}/data/val/"
        output_path = f"s3://{config['S3_BUCKET_NAME']}/outputs"
        instance_type = instance_type
        use_spot_instances = True
        max_run = 3600
        if use_spot_instances:
            max_wait = 7200
        else:
            max_wait = None
        checkpoint_s3_uri = f"s3://{config['S3_BUCKET_NAME']}/checkpoints"
        base_job_name = base_job_name

    sagemaker_run_config = {
        "sagemaker_session": sagemaker_session,
        "role_arn": role_arn,
        "training_input_path": training_input_path,
        "validation_input_path": validation_input_path,
        "output_path": output_path,
        "instance_type": instance_type,
        "use_spot_instances": use_spot_instances,
        "max_run": max_run,
        "max_wait": max_wait,
        "checkpoint_s3_uri": checkpoint_s3_uri,
        "base_job_name": base_job_name,
    }

    return sagemaker_run_config


def generate_wandb_api_key():
    """Login to Weights and Biases and generate secrets.env key file for SageMaker jobs."""
    # Login to W&B
    login = wandb.login()
    if login:
        logging.info("INFO: Already logged into Weights and Biases...")

    # Export secrets.env file for SageMaker
    wandb.sagemaker_auth(path=f"{PROJECT_ROOT_PATH}/sagemaker_src")

    logging.info(
        f"INFO: Weights and Biases secret API key saved to {PROJECT_ROOT_PATH}/sagemaker_src/secrets.env"
    )


# Note the function below was copied from Weights and Biases: https://docs.wandb.ai/guides/track/public-api-guide
def get_wandb_project_runs(
    wandb_user: str, project: Optional[str] = "brain-mri-segmentation"
) -> pd.DataFrame:
    """Retrieves the run data from W&B project into Pandas DataFrame.

    Args:
        wandb_user (str): Your Weights and Biases account username
        project (Optional[str], optional): Name of W&B project to fetch run data from. Defaults to 'brain-mri-segmentation'.

    Returns:
        pd.DataFrame: Pandas DataFrame containing project run information.
    """
    api = wandb.Api()
    runs = api.runs(wandb_user + "/" + project)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    return runs_df


def upload_model_to_s3(
    bucket_name: str,
    local_model_path: Union[str, pathlib.Path],
    key_prefix: Optional[str] = "optuna-model",
) -> str:
    """upload local model file to S3 bucket.

    Args:
        bucket_name (str): Name of S3 bucket where the model will be stored
        local_model_path (Union[str, pathlib.Path]): Path to model file to be uploaded to S3.
        key_prefix (str, optional): Prefix to add before file name. Defaults to 'optuna-model'.

    Returns:
        str: Path to S3 file created.
    """
    sagemaker_session = sagemaker.Session(default_bucket=bucket_name)
    sagemaker_session.default_bucket(), sagemaker_session._region_name
    bucket_model_path = sagemaker_session.upload_data(
        path=local_model_path, bucket=bucket_name, key_prefix=key_prefix
    )
    return bucket_model_path
