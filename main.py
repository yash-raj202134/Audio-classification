# Testing
import sys
import os
from src.audioclf.logger import logging
from src.audioclf.exception import CustomException

from src.audioclf.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.audioclf.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.audioclf.pipeline.stage_03_datatransformation import DataTransformationTrainingPipeline

# logging.info("testing log")

# try:
#     1/0
# except Exception as e:
#     raise CustomException(e,sys)

# STAGE_NAME = "Data ingestion stage"
# try:
#     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     raise CustomException(e,sys)


# STAGE_NAME = "Data Validation stage"

# try:
#     logging.info(f">>>>>> stage {STAGE_NAME} started<<<<<<")
#     data_validation = DataValidationTrainingPipeline()
#     data_validation.main()
#     logging.info(f">>>>>> stage {STAGE_NAME} completed<<<<<<\n\nx==============x")

# except Exception as e:
#     logging.exception(e)
#     raise e 


STAGE_NAME = "Data Transformation stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed<<<<<<\n\nx==============x")

except Exception as e:
    raise CustomException(e,sys)
