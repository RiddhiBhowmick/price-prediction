import os
import sys
from dataclasses import dataclass

import pandas as pd 
import numpy as np 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

# from src.components.data_ingestion import DataIngestion




@dataclass
class DataTransoformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    logging.info("data transformation  path config done")
 
class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransoformationConfig()
    
    def get_data_transformer_obj(self):

        try:

            df=pd.read_csv("D:\ml projects\car price prediction\data\diamonds.csv")  
            """reading the raw data saved in artifacts"""
            
            x=df.drop(['price'],axis=1)

            cat_cols=x.select_dtypes(exclude='number')
            num_cols=x.select_dtypes(include='number')

            
            num_pipeline=Pipeline(steps=[
                ("standard_scaling",StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ("onehotencoding",OneHotEncoder())
            ])
            
            preprocessor=ColumnTransformer(
                [
                ("numerical_columns_transformed",num_pipeline,num_cols.columns.tolist()),
                ("categorical_columns_transformed",cat_pipeline,cat_cols.columns.tolist())
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test part")

            preprocessor_obj=self.get_data_transformer_obj()
            logging.info("got preprocessor obj")

            target_col='price'
            
            """train part"""
            input_feature_train_df=train_df.drop([target_col],axis=1)
            target_feature_train_df=train_df[target_col]

            "test part"
            input_feature_test_df=test_df.drop([target_col],axis=1)
            target_feature_test_df=test_df[target_col]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            logging.info("preprocessing obj applied to train and test feature")


            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("train array and test array created")

            save_obj(
                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )


            return(
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path
                )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    
    data_ingestion_obj=DataIngestion()
    train_path,test_path=data_ingestion_obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_ =data_transformation.initiate_data_transformation(train_path,test_path)