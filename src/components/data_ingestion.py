import os 
import sys

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train_data.csv")
    test_data_path=os.path.join('artifacts',"test_data.csv")
    raw_data_path=os.path.join('artifacts',"raw_data.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("initiated data ingestion part")

        try:
            df=pd.read_csv(r'D:\ml projects\car price prediction\data\diamonds.csv')
            logging.info("read the data from source")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,header=False,index=False)

            train,test=train_test_split(df,random_state=10,shuffle=True,test_size=0.2)
            train.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
        
            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_path,test_path)