import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from  dataclasses import dataclass
import os

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_transformer_obj(self):
        logging.info("Data Transformation initiated")
        
        column=['cleaned']
        
        try:
            pipeline=Pipeline(
                steps=[
                    ("vectorizer",TfidfVectorizer(ngram_range=(1, 2), max_features=5000))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    (pipeline,column)
                ]
            )
            
            logging.info("Data Transformation is completed")
            
            return preprocessor
            
        except Exception as e:
            logging.info(e)
            return CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj=self.get_transformer_obj()
            
            target_column_name="Sentiment"
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file,
            )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__=="__main__":
    obj=DataTransformation()
    obj.initiate_data_transformation()