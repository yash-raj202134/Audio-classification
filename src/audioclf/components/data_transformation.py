import os 
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.audioclf.logger import logging
from src.audioclf.exception import CustomException
from src.audioclf.utils import save_json
from dataclasses import dataclass



from src.audioclf.entity.config_entity import DataTransformationConfig




class DataTransformation:
    def __init__(self,config:DataTransformationConfig) -> None:
        self.config = config


    
    def features_extractor(self,file):


        audio, sample_rate = librosa.load( file, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        
        return mfccs_scaled_features
    
    def initiate_feature_extraction(self):
        ### Now we iterate through every audio file and extract features 
        ### using Mel-Frequency Cepstral Coefficients
        extracted_features=[]
        meta_data = pd.read_csv(self.config.meta_data_path)

        for index_num,row in tqdm(meta_data.iterrows()):
            file_name = os.path.join(os.path.abspath(self.config.audio_data_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
            final_class_labels=row["class"]

            data = self.features_extractor(file_name)
            extracted_features.append([data,final_class_labels])
        
        extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','class'])
        # extracted_features_df

        # saving the file in json format 
        save_json(filename='preprocessed_data.json',data= extracted_features_df,path=self.config.root_dir,)



        

    # def convert_examples_to_features(self,example_batch):
    #     input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )
        
    #     with self.tokenizer.as_target_tokenizer():
    #         target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )
            
    #     return {
    #         'input_ids' : input_encodings['input_ids'],
    #         'attention_mask': input_encodings['attention_mask'],
    #         'labels': target_encodings['input_ids']
    #     }
    # def convert(self):
    #     dataset_samsum = load_from_disk(self.config.data_path)
    #     dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features,batched= True)

    #     dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
    