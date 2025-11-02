from src.logger.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    Ingestion = DataIngestion()
    train_data_path,test_data_path = DataIngestion().initiate_data_ingestion()
    logging.info(f'Data Ingestion completed {train_data_path,test_data_path}')
    print(f'Data Ingestion competed {train_data_path,test_data_path}')

    transform = DataTransformation()
    train_arr,test_arr,_ = transform.initiate_data_transformation(train_data_path,test_data_path)
    logging.info(f'Data Transformation completed {train_arr,test_arr}')
    print(f'Data Transformation completed {train_arr,test_arr}')