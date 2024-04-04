# Entity
from dataclasses import dataclass , field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE: str
    ALL_REQUIRED_FOLDERS: List[Path] =  field(default_factory=list)



@dataclass
class DataTransformationConfig:
    root_dir:Path
    audio_data_path: str
    meta_data_path: str


@dataclass
class ModelTrainerConfig:
    root_dir:Path
    preprocessed_data_path: str
    trained_model_path: str
    num_train_epochs: int
    per_device_train_batch_size: int