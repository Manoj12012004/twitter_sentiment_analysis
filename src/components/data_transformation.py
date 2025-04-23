import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import sys
from src.exception import CustomException
from src.logger import logging

from  dataclasses import dataclass
import os

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    
    def initiate_data_transformation(self):
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
    
if __name__=="__main__":
    obj=DataTransformation()
    obj.initiate_data_transformation()