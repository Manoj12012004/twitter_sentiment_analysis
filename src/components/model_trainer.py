import sys
import os
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import time

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_obj = ModelTrainerConfig()

    def evaluate_model(self, X_train, X_test, y_train, y_test, models):
        try:
            report = {}
            
            for model_name, model in models.items():
                try:
                    start = time.time()
                    model.fit(X_train, y_train)
                    duration = time.time() - start
                    logging.info(f"{name} trained in {duration:.2f} seconds")

                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    train_model_score = accuracy_score(y_train, y_train_pred)
                    test_model_score = accuracy_score(y_test, y_test_pred)

                    report[model_name] = test_model_score
                except Exception as e:
                    CustomException(e,sys)
            return report
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_train(self, train_arr, test_arr):
        try:
            logging.info("Splitting the dataset for train and test")
            train_arr=train_arr.toarray()
            test_arr=test_arr.toarray()
            X_train = train_arr[:, :5000]
            y_train = train_arr[:, 5000]

            X_test = test_arr[:, :5000]
            y_test = test_arr[:, 5000]
            logging.info(X_train)
            logging.info(y_train)
            
            
            X_train = csr_matrix(X_train)
            X_test = csr_matrix(X_test)
            logging.info(X_train)
            try:
                models = {
                    'LogisticRegression': LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1),
                    # Add more models here as needed
                }

                model_report = self.evaluate_model(X_train, X_test, y_train, y_test, models)

                best_model_name = max(model_report, key=model_report.get)
                best_model_score = model_report[best_model_name]
                best_model = models[best_model_name]

                if best_model_score < 0.6:
                    raise CustomException('No model performed well (score < 0.6)')

                logging.info(f'Best model found: {best_model_name} with score {best_model_score}')

                save_object(file_path=self.model_obj.model_file_path, obj=best_model)

                predicted = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, predicted)

                logging.info(f'Model accuracy on test data: {accuracy}')
                return accuracy
            except:
                logging('hi')

        except Exception as e:
            raise CustomException(e, sys)
