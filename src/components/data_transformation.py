import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
from dataclasses import dataclass
import re

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def clean_text(self,text):
        text=re.sub(r'\@w+|\#','', text)
        text=re.sub(r'[^A-Za-z\s]',"",text)
        text=text.lower()
        return text
    def get_transformer_obj(self):
        logging.info("Data Transformation initiated")

        column = 'Cleaned'  # Assuming 'Cleaned' is the column containing the text data

        try:
            # Define the pipeline to apply TfidfVectorizer
            pipeline = Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=5000))
                ]
            )

            # Apply the pipeline inside a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('pipeline', pipeline, column)  # Apply pipeline to 'Cleaned' column
                ]
            )

            logging.info("Data Transformation is completed")


            return preprocessor

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(f"Error: {e}", sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Define column names
            col = ['ID', 'Entity', 'Sentiment', 'Content']
            
            # Load and clean the training data
            train_df = pd.read_csv(train_path, names=col)
            train_df = train_df.dropna()  # Drop rows with missing values
            
            # Apply text cleaning
            train_df['Cleaned'] = train_df['Content'].apply(self.clean_text)
            
            # Remove stopwords from the 'Cleaned' column
            stop = set(stopwords.words('english'))
            train_df['Cleaned'] = train_df['Cleaned'].apply(lambda x: ' '.join([w for w in x.split(' ') if w not in stop]))
            
            # Log the cleaned text for inspection
            logging.info(f"Sample cleaned text: {train_df['Cleaned'].head()}")
            
            # Convert 'Sentiment' to category and add a numerical label
            train_df['Sentiment'] = train_df['Sentiment'].astype('category')
            train_df['S_label'] = train_df['Sentiment'].cat.codes
            
            logging.info(train_df.head())
            logging.info(f"Missing values: {train_df.isna().sum()}")

            test_df = pd.read_csv(test_path, names=col)
            test_df = test_df.dropna()  # Drop rows with missing values
            
            # Apply text cleaning
            test_df['Cleaned'] = test_df['Content'].apply(self.clean_text)
            
            # Remove stopwords from the 'Cleaned' column
            stop = set(stopwords.words('english'))
            test_df['Cleaned'] = test_df['Cleaned'].apply(lambda x: ' '.join([w for w in x.split(' ') if w not in stop]))
            
            # Log the cleaned text for inspection
            logging.info(f"Sample cleaned text: {test_df['Cleaned'].head()}")
            
            # Convert 'Sentiment' to category and add a numerical label
            test_df['Sentiment'] = test_df['Sentiment'].astype('category')
            test_df['S_label'] = test_df['Sentiment'].cat.codes
            
            logging.info(test_df.head())
            logging.info(f"Missing values: {test_df.isna().sum()}")
            
            # Get the preprocessor object (ColumnTransformer)
            p_obj = self.get_transformer_obj()
            
            input_feature_train_df=train_df
            input_feature_test_df=test_df

            target_feature_train_df=train_df['S_label']
            target_feature_test_df=test_df['S_label']
            # Apply the transformation on the 'Cleaned' column
            input_feature_train_arr=p_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=p_obj.transform(input_feature_test_df)

            logging.info(f"Transformed feature matrix shape: {input_feature_train_arr}")
            
            train_arr = np.c_[input_feature_train_arr.toarray().astype(np.float32), np.array(target_feature_train_df).reshape(-1, 1).astype(np.float32)]
            test_arr = np.c_[input_feature_test_arr.toarray().astype(np.float32), np.array(target_feature_test_df).reshape(-1,1).astype(np.float32)]
            

            # Optionally save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file, p_obj)
            logging.info("Saved preprocessor")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file,
            )
            
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(f"Error: {e}", sys)