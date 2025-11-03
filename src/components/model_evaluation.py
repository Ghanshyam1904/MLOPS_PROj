import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle
from src.util.util import load_object
from src.logger.logger import logging
from src.exception.exception import CustomException

from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

class ModelEvaluation:
    def __init__(self):
        logging.info('Evaluation started')


    def evalmetrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        logging.info('evaluation metrics captured')
        return rmse,mae,r2

    def initiate_model_evaluation(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependant and In-Dependant Variable from train and test')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            model_path = os.path.join('artifacts','model.pkl')
            model = load_object(model_path)
            # Registered the Model to MLFLOW
            #mlflow.set_registry_uri("")
            logging.info('Model have registered')
            tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type)

            with mlflow.start_run():
                prediction = model.predict(X_test)
                (rmse , mae,r2) = self.evalmetrics(y_test,prediction)

                mlflow.log_metric('rmse',rmse)
                mlflow.log_metric('mae',mae)
                mlflow.log_metric('r2',r2)

                # Model registry does not work with file store
                if tracking_url_type != 'file':
                    # Register the model
                    # There are other ways ti use model registry
                    # please refer to the doc of more information
                    mlflow.sklearn.log_model(model,'model',registered_model_name='ml_model')
                else:
                    mlflow.sklearn.log_model(model,'model')

        except Exception as e:
            logging.info('Error in Model Evaluation ',e)
            raise CustomException(e,sys)