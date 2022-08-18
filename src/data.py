import getpass
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from typing import List, Union
from zipfile import ZipFile

import sagemaker
import yaml
from natsort import natsorted
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("./../config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

kaggle_token_path = os.path.join(
    os.path.expanduser("~"), ".kaggle/kaggle.json"
)


def create_kaggle_token_file() -> None:
    """
    Creates a restricted file containing users Kaggle API details. When executed the
    user is asked to enter there Kaggle username and API token. These details are then stored in
    the users home directory in a file named './kaggle/kaggle.json'.
    """
    if not os.path.isfile(kaggle_token_path):
        # Create the missing Kaggle token file with restricted permissions
        logger.info(f"Creating Kaggle token file at {kaggle_token_path}")
        subprocess.run(["mkdir", "-p", os.path.dirname(kaggle_token_path)])
        subprocess.run(["touch", kaggle_token_path])
        subprocess.run(["chmod", "600", kaggle_token_path])
    if os.path.getsize(kaggle_token_path) == 0:
        # Store Kaggle username and API token
        kaggle_username = input("Please enter Kaggle username: ")
        kaggle_token = getpass.getpass(
            prompt="Please enter Kaggle API token: "
        )
        with open(
            os.path.join(os.path.expanduser("~"), ".kaggle/kaggle.json"), "w"
        ) as f:
            f.write(
                json.dumps({"username": kaggle_username, "key": kaggle_token})
            )
        logger.info(f"Kaggle token file saved in {kaggle_token_path}")
    else:
        logger.info(
            f"Kaggle token file already created in {kaggle_token_path}."
        )


def download_kaggle_dataset(
    kaggle_dataset: str, output_path: Union[str, pathlib.Path]
) -> None:
    """Downloads dataset specified by kaggle_dataset from Kaggle.

    Args:
        kaggle_dataset (str): Kaggle dataset to download
        output_path (Union[str, pathlib.Path]): Path where downloaded files should be saved.
    """
    if (
        os.path.isfile(kaggle_token_path)
        and os.path.getsize(kaggle_token_path) != 0
    ):
        subprocess.run(["mkdir", "-p", output_path])
        logger.info(
            f"Starting download of {kaggle_dataset} dataset from Kaggle."
        )
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                kaggle_dataset,
                "-p",
                output_path,
            ]
        )
        logger.info(
            f"Finished downloading data. Data downloaded to {output_path}"
        )
    else:
        logger.error(
            "Kaggle token file is missing. Please use the create_kaggle_token_file function first."
        )
        sys.exit()


def extract_kaggle_dataset(
    file_dir: Union[str, pathlib.Path], filename: str
) -> None:
    """Extracts contents from zipped file downloaded fom Kaggle.

    Args:
        file_dir (Union[str, pathlib.Path]): Path of directory containing zip file with data.
        filename (str): Name of zip file.
    """
    file_path = os.path.join(file_dir, filename)
    with ZipFile(file_path, "r") as zipObj:
        # Extract all zip file contents
        logging.info(f"Starting extraction of data stored in {file_path}.")
        zipObj.extractall(path=file_dir)
        logger.info(f"Dataset extracted in {file_dir}")


def move_data_folders(
    dataset_paths: List[Union[str, pathlib.Path]],
    output_path: Union[str, pathlib.Path],
) -> None:
    """Moves data folders to new location.

    This function is used to move data folders contained in dataset_paths to a new location,
    such as splitting data for training, validation and testing.

    Args:
        dataset_paths (List[Union[str, pathlib.Path]]): A list of folder paths for the data to be moved.
        output_path (Union[str, pathlib.Path]): Path where data should be moved to.
    """
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
        logging.info(f"Created {output_path} directory.")
    for folder in dataset_paths:
        shutil.move(folder, output_path)
        logging.info(f"Moved {folder} to {output_path}.")


def extract_patient_filepaths(
    patient_data_path: Union[str, pathlib.Path]
) -> List[dict]:
    """Output list of dictionaries for patient MRI and mask data.

    Args:
        patient_data_path (Union[str, pathlib.Path]): Path containing patient data.

    Returns:
        List[dict]: List of dictionariees with path to input MRI image and corresponding segmentation mask.
    """

    all_files = []
    for root, folders, files in os.walk(patient_data_path):
        for file in natsorted(files):
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


def extract_dataset_filepaths(
    data_dir_paths: List[Union[str, pathlib.Path]]
) -> List[dict]:
    """Returns a list of dictionaries with corresponding image and mask file paths for an entire dataset.

    Args:
        data_dir_paths (List[Union[str, pathlib.Path]]): List containing paths to patient data.

    Returns:
        List[dict]: A list of dictionaries with all image and mask file paths contained in the input data_dir_paths.
    """
    all_filepaths = []
    for dir in data_dir_paths:
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith("_mask.tif") is False:
                    file_mask = file[:-4] + "_mask.tif"
                    if file_mask in files:
                        all_filepaths.append(
                            {
                                "image": f"{root}/{file}",
                                "mask": f"{root}/{file_mask}",
                            }
                        )
    return all_filepaths


def setup_brain_mri_dataset(data_root_path: Union[str, pathlib.Path]) -> None:
    """Downloads and creates datasets from Kaggle Brain MRI dataset.

    This function downloads, extracts and sorts the Kaggle Brain MRI data into train, val and test datasets.

    Args:
        data_root_path (Union[str, pathlib.Path]): Path where data will be stored.
    """
    kaggle_dataset = config["KAGGLE_DATASET"]
    zip_filename = config["DATA_ZIP_FILENAME"]
    zipped_data_path = os.path.join(data_root_path, zip_filename)

    if not os.path.isfile(zipped_data_path):
        download_kaggle_dataset(
            kaggle_dataset=kaggle_dataset, output_path=data_root_path
        )
        extract_kaggle_dataset(file_dir=data_root_path, filename=zip_filename)
    else:
        logging.info("Data already downloaded from Kaggle. Skipping download.")

    if not os.path.isdir(
        os.path.join(data_root_path, "brain-mri-dataset/train")
    ):
        # Split patients into training, validation and test folders
        logging.info(
            "Starting splitting of data into train, val and test datasets."
        )
        brain_mri_dataset_path = os.path.join(
            data_root_path, "brain-mri-dataset"
        )
        subprocess.run(["mkdir", "-p", brain_mri_dataset_path])
        extracted_data_path = os.path.join(
            data_root_path, "lgg-mri-segmentation/kaggle_3m"
        )
        subfolders = [
            f.path for f in os.scandir(extracted_data_path) if f.is_dir()
        ]
        train_patient_paths, test_patient_paths = train_test_split(
            subfolders, test_size=0.2, random_state=1
        )
        val_patient_paths = test_patient_paths[0:11]
        test_patient_paths = test_patient_paths[11:]

        for dataset_paths, dataset_name in zip(
            [train_patient_paths, val_patient_paths, test_patient_paths],
            ["train", "val", "test"],
        ):
            output_path = os.path.join(brain_mri_dataset_path, dataset_name)
            move_data_folders(
                dataset_paths=dataset_paths, output_path=output_path
            )
        logging.info(
            "Finished splitting into train, validation and test datasets."
        )
    else:
        logging.info(
            "Training, validation and test datasets already prepared. Skipping step."
        )


def upload_data_to_s3(bucket_name, local_dataset_path, key_prefix="data"):
    sagemaker_session = sagemaker.Session(default_bucket=bucket_name)
    sagemaker_session.default_bucket(), sagemaker_session._region_name
    try:
        num_files = len(
            sagemaker_session.list_s3_files(
                bucket=bucket_name, key_prefix=key_prefix
            )
        )
        if num_files == 7858:
            logging.info(
                f"Data already uploaded to s3://{bucket_name}/{key_prefix}. Skipping this step."
            )
        else:
            logging.info(
                f"Expected 7858 files, but found {num_files}. Starting data upload."
            )
            bucket_data_path = sagemaker_session.upload_data(
                path=local_dataset_path, bucket=bucket_name, key_prefix="data"
            )
            logging.info(f"Data has been uploaded to: {bucket_data_path}")
    except:  # BETTER TO PUT SPECIFIC NoSuchBucket error, but not sure how?
        logging.info(
            f"Bucket named {bucket_name} doesnt exist. Starting upload to S3"
        )
        bucket_data_path = sagemaker_session.upload_data(
            path=local_dataset_path, bucket=bucket_name, key_prefix="data"
        )
        logging.info(f"Data has been uploaded to: {bucket_data_path}")
