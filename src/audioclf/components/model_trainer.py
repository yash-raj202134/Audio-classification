import os 
import sys
from src.audioclf.logger import logging
from src.audioclf.exception import CustomException

from src.audioclf.entity.config_entity import ModelTrainerConfig
from src.audioclf.utils import save_model
import matplotlib.pyplot as plt
# model trainer imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
from sklearn import metrics
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):

        # Loading the preprocessed dataset
        extracted_features = pd.read_json(self.config.preprocessed_data_path)

        ### Split the dataset into independent and dependent dataset
        X=np.array(extracted_features['feature'].tolist())
        y=np.array(extracted_features['class'].tolist())

        # encoding the target variable
        labelencoder=LabelEncoder()
        y=to_categorical(labelencoder.fit_transform(y))

        ## Train test split 
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
        ### No of classes
        num_labels=y.shape[1]

        # Building a base model:
        model=Sequential()
        ###first layer
        model.add(Dense(100,input_shape=(40,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        ###second layer
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        ###third layer
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        ###final layer
        model.add(Dense(num_labels))
        model.add(Activation('softmax'))

        print(model.summary())


        # compiling the model:
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')



        # Training my model:

        # trainer_args = (
        #     output_dir = self.config.root_dir, num_train_epochs=self.config.num_train_epochs,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size
        # ) 

        try:
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.config.trained_model_path,"model.hdf5"), 
                                verbose=1, save_best_only=True)
            
            start = datetime.now()

            history = model.fit(X_train, y_train, 
                    batch_size=self.config.per_device_train_batch_size,
                    epochs=self.config.num_train_epochs,
                    validation_data=(X_test, y_test), 
                    callbacks=[checkpointer], 
                    verbose=1
                    )
            
            # Plot training and validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

            # Save the plot to a file
            loss_plot_file = os.path.join(self.config.root_dir, 'loss_plot.png')
            plt.savefig(loss_plot_file)
            plt.close()  # Close the plot to release memory


            # Plot training and validation accuracy
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')

            # Save the plot to a file
            accuracy_plot_file = os.path.join(self.config.root_dir, 'accuracy_plot.png')
            plt.savefig(accuracy_plot_file)
            plt.close()  # Close the plot to release memory
            

            duration = datetime.now() - start
            print("Training completed in time: ", duration)

        except Exception as e:
            raise CustomException(e,sys)
        
        # test accuracy:

        test_accuracy=model.evaluate(X_test,y_test,verbose=0)
        print(f"Accuracy : {test_accuracy[1]}")

        # saving the best model:
        save_model(model=model,filename=os.path.join(self.config.trained_model_path,'best_model.hdf5'))
        logging.info(f"Model training details : epoch_size :{self.config.num_train_epochs},batch_size={self.config.per_device_train_batch_size} accuracy: {test_accuracy}")
