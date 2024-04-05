
import pandas as pd
import numpy as np
from tqdm import tqdm
import os,sys

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
        X=np.array(test_data['feature'].tolist())
        y=np.array(test_data['class'].tolist())



        X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)
        labelencoder=LabelEncoder()
        y=to_categorical(labelencoder.fit_transform(y))

        test_accuracy=model.evaluate(X_test,y_test,verbose=0)
        print(f"Accuracy : {test_accuracy[1]}")





    # def generate_batch_sized_chunks(self,list_of_elements, batch_size):
    #     """split the dataset into smaller batches that we can process simultaneously
    #     Yield successive batch-sized chunks from list_of_elements."""
    #     for i in range(0, len(list_of_elements), batch_size):
    #         yield list_of_elements[i : i + batch_size]

    
    # def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer, 
    #                            batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
    #                            column_text="article", 
    #                            column_summary="highlights"):
    #     article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
    #     target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

    #     for article_batch, target_batch in tqdm(
    #         zip(article_batches, target_batches), total=len(article_batches)):
            
    #         inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
    #                         padding="max_length", return_tensors="pt")
            
    #         summaries = model.generate(input_ids=inputs["input_ids"].to(device),
    #                         attention_mask=inputs["attention_mask"].to(device), 
    #                         length_penalty=0.8, num_beams=8, max_length=128)
    #         ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
            
    #         # Finally, we decode the generated texts, 
    #         # replace the  token, and add the decoded texts with the references to the metric.
    #         decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
    #                                 clean_up_tokenization_spaces=True) 
    #             for s in summaries]      
            
    #         decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            
    #         metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
    #     #  Finally compute and return the ROUGE scores.
    #     score = metric.compute()
    #     return score


    # def evaluate(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
    #     model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
    #     #loading data 
    #     dataset_samsum_pt = load_from_disk(self.config.data_path)


    #     rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
  
    #     rouge_metric = load_metric('rouge')

    #     score = self.calculate_metric_on_test_ds(
    #     dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
    #         )

    #     rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

    #     df = pd.DataFrame(rouge_dict, index = ['pegasus'] )
    #     df.to_csv(self.config.metric_file_name, index=False)
