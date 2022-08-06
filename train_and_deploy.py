import logging

import yaml

from src.data import create_kaggle_token_file, setup_brain_mri_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("./config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    # PREPARE DATASET
    create_kaggle_token_file()
    setup_brain_mri_dataset(data_root_path=config["DATA_ROOT_PATH"])


if __name__ == "__main__":
    main()
