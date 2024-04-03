# from cnnClassifier.config.configuration import ConfigurationManager
# from cnnClassifier.components.data_ingestion import DataIngestion
import sys,os
from src.audioclf.config.configuration import ConfigurationManager
from src.audioclf.components.data_ingestion import DataIngestion


from src.audioclf.logger import logging
from src.audioclf.exception import CustomException


STAGE_NAME = "Data ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e
        


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e,sys)