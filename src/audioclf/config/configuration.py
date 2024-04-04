# Configuration
from src.audioclf.constants import *
from src.audioclf.utils import read_yaml, create_directories

# from cnnClassifier.entity.config_entity import DataIngestionConfig , PrepareBaseModelConfig ,TrainingConfig , EvaluationConfig
from src.audioclf.entity.config_entity import (DataIngestionConfig ,DataValidationConfig ,
DataTransformationConfig ,ModelTrainerConfig)

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
    
    

    # def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
    #     config = self.config.prepare_base_model
        
    #     create_directories([config.root_dir])

    #     prepare_base_model_config = PrepareBaseModelConfig(
    #         root_dir=Path(config.root_dir),
    #         base_model_path=Path(config.base_model_path),
    #         updated_base_model_path=Path(config.updated_base_model_path),
    #         params_image_size=self.params.IMAGE_SIZE,
    #         params_learning_rate=self.params.LEARNING_RATE,
    #         params_include_top=self.params.INCLUDE_TOP,
    #         params_weights=self.params.WEIGHTS,
    #         params_classes=self.params.CLASSES
    #     )

    #     return prepare_base_model_config
    
    # def get_training_config(self) -> TrainingConfig:
    #     training = self.config.training
    #     prepare_base_model = self.config.prepare_base_model
    #     params = self.params
    #     training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
    #     create_directories([
    #         Path(training.root_dir)
    #     ])

    #     training_config = TrainingConfig(
    #         root_dir=Path(training.root_dir),
    #         trained_model_path=Path(training.trained_model_path),
    #         updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
    #         training_data=Path(training_data),
    #         params_epochs=params.EPOCHS,
    #         params_batch_size=params.BATCH_SIZE,
    #         params_is_augmentation=params.AUGMENTATION,
    #         params_image_size=params.IMAGE_SIZE
    #     )

    #     return training_config
    # def get_evaluation_config(self) -> EvaluationConfig:
    #     eval_config = EvaluationConfig(
    #         path_of_model="artifacts/training/model.h5",
    #         training_data="artifacts/data_ingestion/kidney-ct-scan-image",
    #         mlflow_uri="https://dagshub.com/yash-raj202134/Kidney-Disease-Classification-deep-learning.mlflow",
    #         all_params=self.params,
    #         params_image_size=self.params.IMAGE_SIZE,
    #         params_batch_size=self.params.BATCH_SIZE
    #     )
    #     return eval_config


