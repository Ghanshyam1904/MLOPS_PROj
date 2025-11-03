from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.exception.exception import CustomException
from src.logger.logger import logging

from src.util.util import save_object
from src.util.util import evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting Dependant and In-Dependant Variable from train and test')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print('\n')
            print(y_train)
            logging.info('done12')
            models = {
                "Linear Regression ": LinearRegression(),
                "Lasso ": Lasso(),
                "Ridge ": Ridge(),
                "Elastic Net ": ElasticNet(),
                "Random Forest ": RandomForestRegressor(),
                "xgboost":XGBRegressor()
            }
            model_report:dict=evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n====================================================")
            logging.info(f"Model Report : {model_report}")

            # To get the best model from the dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name}, R2 Score : {best_model_score}')
            print("\n================================================================================")
            logging.info(f"Best Model Found, Model Name is : {best_model_name} , R2 Score is : {best_model_score}")


            # Always persist the selected model locally for serving
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.info('Error in Model Evaluation ',e)
            raise CustomException(e,sys)