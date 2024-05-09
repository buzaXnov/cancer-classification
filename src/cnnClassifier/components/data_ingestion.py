import os
import zipfile

import gdown

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig

# from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_data(self):
        """Downloads the dataset as a zip file from Google Drive."""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(
                f"Downloading data from {dataset_url} into file {zip_download_dir}."
            )

            file_id = dataset_url.split("/")[-2]

            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(url=prefix + file_id, output=zip_download_dir)

            logger.info(
                f"Downloaded data from {dataset_url} into file {zip_download_dir}"
            )

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """Extracs the zip file into the data directory."""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(path=unzip_path)
