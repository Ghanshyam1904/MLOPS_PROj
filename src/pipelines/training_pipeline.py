from src.logger.logger import logging
from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    Ingestion = DataIngestion()
    train_data_path,test_data_path = DataIngestion().initiate_data_ingestion()
    logging.info(f'Data Ingestion completed {train_data_path,test_data_path}')
    print(f'Data Ingestion competed {train_data_path,test_data_path}')