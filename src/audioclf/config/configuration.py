# Configuration
from src.audioclf.constants import *
from src.audioclf.utils import read_yaml, create_directories

from src.audioclf.entity.config_entity import (DataIngestionConfig ,DataValidationConfig ,
DataTransformationConfig ,ModelTrainerConfig,ModelEvaluationConfig)

import os
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self)->DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            ALL_REQUIRED_FOLDERS = config.ALL_REQUIRED_FOLDERS,

        )

        return data_validation_config
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            audio_data_path=config.audio_data_path,
            meta_data_path=config.meta_data_path,

        )

        return data_transformation_config
    

    def get_model_trainer_config(self)->ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])


        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            preprocessed_data_path = config.preprocessed_data_path,
            trained_model_path=config.trained_model_path,
            num_train_epochs= params.num_train_epochs,
            per_device_train_batch_size = params.per_device_train_batch_size,
            
        )
        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.TestingArguments

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path = config.model_path,
            metric_file_name = config.metric_file_name,
            test_data_size = params.test_data_size
           
        )

        return model_evaluation_config


