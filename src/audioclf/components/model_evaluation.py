
import pandas as pd
import numpy as np
from tqdm import tqdm
import os,sys
import yaml


from src.audioclf.exception import CustomException
from src.audioclf.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.audioclf.entity.config_entity import ModelEvaluationConfig
from src.audioclf.utils import load_saved_model


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def prepare_test_dataset(self,data,test_size):
        num_samples = len(data)
        if isinstance(test_size, float):
            test_size = int(test_size * num_samples)

        indices = np.random.permutation(num_samples)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]


        return train_data,test_data


    def evaluate(self):

        '''input -> test_data : output ->evaluation_metrics.csv'''

        # loading dataset:

        dataset = pd.read_json(self.config.data_path)


        # print(dataset)

        model = load_saved_model(self.config.model_path)
        # print(dataset.shape)

        train_data,test_data = self.prepare_test_dataset(data=dataset,test_size=self.config.test_data_size)


        ### Split the dataset into independent and dependent dataset
        X = np.array(test_data['feature'].tolist())
        y = np.array(test_data['class'].tolist())

        labelencoder=LabelEncoder()
        y = to_categorical(labelencoder.fit_transform(y))



        X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)
        ### No of classes
        num_labels=y.shape[1]
        

        test_accuracy=model.evaluate(X_test,y_test,verbose=0)
        print(f"Accuracy : {test_accuracy[1]}")

        # Generate classification report
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        report = classification_report(np.argmax(y_test, axis=-1), y_pred, output_dict=True)

         # Convert report to DataFrame
        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(os.path.join(self.config.root_dir,"evaluation.csv"))


        # Write data to YAML file
        with open(os.path.join(self.config.root_dir,"evaluation.yaml"), 'a') as yaml_file:
            yaml.dump(report, yaml_file)
