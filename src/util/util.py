import os
import sys
import pickle

from sklearn.metrics import r2_score

from src.exception.exception import CustomException
from src.logger.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

            logging.info(f'Model: {model_name}, R2 Score: {test_model_score}')

        return report
    except Exception as e:
        logging.error('Error in Evaluate model')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Error in Load model in utils')
        raise CustomException(e,sys)