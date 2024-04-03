
from src.audioclf.config.configuration import ConfigurationManager
from src.audioclf.components.data_transformation import DataTransformation
from src.audioclf.logger import logging
from src.audioclf.exception import CustomException


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_feature_extraction()