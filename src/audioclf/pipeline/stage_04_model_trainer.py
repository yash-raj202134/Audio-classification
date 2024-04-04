
from src.audioclf.config.configuration import ConfigurationManager
from src.audioclf.components.model_trainer import ModelTrainer
from src.audioclf.logger import logging
from src.audioclf.exception import CustomException



class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()