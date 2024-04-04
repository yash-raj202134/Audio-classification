
from src.audioclf.config.configuration import ConfigurationManager
from src.audioclf.components.model_evaluation import ModelEvaluation

from src.audioclf.logger import logging
from src.audioclf.exception import CustomException



class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()