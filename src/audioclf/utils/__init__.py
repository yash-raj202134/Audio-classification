# extracting the data (data ingestion)
import tarfile
import os
import sys
import yaml
import json
from box import ConfigBox
import pandas as pd
from pathlib import Path
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from tensorflow.keras.models import load_model

from src.audioclf.logger import logging
from src.audioclf.exception import CustomException


def extract_tar_gz(file_path, extract_path):
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        logging.info("Extraction successful.")

        os.remove('UrbanSound8K.tar.gz')
    except Exception as e:
        raise CustomException(e,sys)




def save_model(model, filename):
    """
    Save a Keras model to an HDF5 file.

    Parameters:
        model (tensorflow.keras.models.Model): The Keras model to save.
        filename (str): The filename (including path) to save the model to.
    """
    model.save(filename)
    print(f"Model saved to '{filename}'")

    

def load_saved_model(filename):
    """
    Load a Keras model from an HDF5 file.

    Parameters:
        filename (str): The filename (including path) to load the model from.

    Returns:
        tensorflow.keras.models.Model: The loaded Keras model.
    """
    model = load_model(filename)
    logging.info(f"Model loaded from '{filename}'")
    return model

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


@ensure_annotations
def save_json(filename,data,path):
    """
    Save a pandas DataFrame to a JSON file.
    Parameters:
        filename (str): The filename (including extension) for the JSON file.
        dataframe (pandas.DataFrame): The DataFrame to save.
        path (str): The path where the file will be saved.
    """
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join(path, filename)

    try:
        data.to_json(file_path)
        logging.info(f"DataFrame saved to '{file_path}' as JSON.")
    except Exception as e:
        logging.error(f"Error occurred while saving DataFrame to '{file_path}': {e}")



# @ensure_annotations
# def load_json(path: Path) -> pd.DataFrame:
#     """load json files data
#     Args:
#         path (Path): path to json file

#     Returns: Dataframe
#     """
#     dataframe = pd.read_json(path)

#     logging.info(f"json file loaded succesfully from: {path}")
#     return dataframe
