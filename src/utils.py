import logging
import os
import sys

import boto3
import git
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
