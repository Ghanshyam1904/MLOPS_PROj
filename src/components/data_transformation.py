from sklearn.preprocessing import OrdinalEncoder ## For categorical cols
from sklearn.preprocessing import StandardScaler ## Scaling down the features
from sklearn.impute import SimpleImputer ## Filling Missing Values
# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
import sys,os
import pandas as pd
import numpy as np

from src.logger.logger import logging
from src.exception.exception import CustomException
from src.util.util import save_object

## Data Transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config =  DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation is Started')
            # Define which cols should be ordinal-encoded and which should be scaled
            num_col = ['carat','depth','table','x','y','z']
            cat_col = ['cut','color','clarity']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline is initiated')
            # Numerical Transformation
            num_trans = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical transformation
            cat_trans = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
            ])

            # merging preprocessor Transformation
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_trans, num_col),
                ('cat', cat_trans, cat_col)
            ])

            return preprocessor
            logging.info('Pipeline is Completed')

        except Exception as e:
            logging.info('Error in Data Tranformation for getting')
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformation_obj()

            target_col_name = 'price'
            drop_col = [target_col_name,'id']

            # Features into in-dependant and dependant
            input_feature_train_df = train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=drop_col, axis=1)
            target_feature_test_df = test_df[target_col_name]

            ## Apply the transformation
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test_df)

            logging.info("Applying preprocessing object on traning and testing data")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessor_obj
            )
            logging.info('Preprocessor pickle is created')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
## Data Ingestion class

