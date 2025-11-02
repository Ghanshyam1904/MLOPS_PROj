import sys
import os
import pandas as pd
from src.logger.logger import logging
from sklearn.model_selection import train_test_split
from src.exception.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingetion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Initiating data ingestion')
        try:
            df = pd.read_csv(os.path.join('experiments\Dataset','gemstone.csv'))

            # Create the folder for raw data path
            os.makedirs(os.path.dirname(self.data_ingetion_config.raw_data_path),exist_ok=True)

            # Save the raw data into csv
            df.to_csv(self.data_ingetion_config.raw_data_path,index=False,header=True)

            logging.info('Training and Splitting the data')
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)

            # store the train set and test set in the artifacts
            train_set.to_csv(self.data_ingetion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingetion_config.test_data_path,index=False,header=True)

            return(
                self.data_ingetion_config.train_data_path,
                self.data_ingetion_config.test_data_path
            )
        except Exception as e:
            logging.info(f'Error in Data Ingestion:{e}')
            CustomException(e,sys)