# Data ingestion
import os , sys
import zipfile
import gdown

from src.audioclf.logger import logging
from src.audioclf.exception import CustomException
from src.audioclf.utils import extract_tar_gz
from src.audioclf.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            gdown.download(dataset_url,zip_download_dir)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")


        except Exception as e:
            raise CustomException(e,sys)
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        extract_tar_gz(file_path=self.config.local_data_file,extract_path=self.config.unzip_dir)
