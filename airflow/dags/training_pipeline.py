# ------------------------------------------------------------
# DAG Name: Gemstone_Training_Pipeline
# Description: This Airflow DAG automates the machine learning
# training pipeline for a Gemstone classification project.
# ------------------------------------------------------------

from __future__ import annotations
import json
from textwrap import dedent
import pendulum
import boto3  # Used for uploading artifacts to AWS S3

# Import Airflow core components
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import the custom training pipeline class from your project
from src.pipelines.TrainingPipeline import TrainingPipeline

# Create an instance of your TrainingPipeline
training_pipeline = TrainingPipeline()

# ------------------------------------------------------------
# DAG Definition
# ------------------------------------------------------------
with DAG(
    dag_id='Gemstone_Training_Pipeline',           # Unique DAG name
    default_args={'retries': 2},                   # Retry failed tasks twice
    description='Gemstone Training Pipeline',      # Description visible in Airflow UI
    schedule_interval='@weekly',                   # Run once per week
    start_date=pendulum.datetime(2025, 11, 4, tz='utc'),  # Start date (UTC timezone)
    catchup=False,                                 # Do not backfill past runs
    tags=['machine learning', 'classification', 'gemstone'],  # Tags for UI filtering
) as dag:
    dag.doc_md = __doc__  # Display this docstring in Airflow UI

    # --------------------------------------------------------
    # Task 1: Data Ingestion
    # --------------------------------------------------------
    def data_ingestion(**kwargs):
        """
        This task reads raw data, splits it into train and test sets,
        and saves them for the next stage.
        """
        ti = kwargs['ti']  # Task instance for XCom communication
        train_data_path, test_data_path = training_pipeline.start_data_ingestion()
        ti.xcom_push('data_ingestion_artifact', {
            'train_data_path': train_data_path,
            'test_data_path': test_data_path
        })

    # --------------------------------------------------------
    # Task 2: Data Transformation
    # --------------------------------------------------------
    def data_transformation(**kwargs):
        """
        This task performs feature engineering and data preprocessing.
        It reads paths from the previous step via XCom.
        """
        import numpy as np
        ti = kwargs['ti']
        data_ingestion_artifact = ti.xcom_pull(
            task_ids='data_ingestion',
            key='data_ingestion_artifact'
        )
        train_arr, test_arr = training_pipeline.start_data_transformation(
            data_ingestion_artifact['train_data_path'],
            data_ingestion_artifact['test_data_path']
        )
        # Convert arrays to lists before pushing to XCom (JSON serializable)
        ti.xcom_push('data_transformation_artifact', {
            'train_arr': train_arr.tolist(),
            'test_arr': test_arr.tolist()
        })

    # --------------------------------------------------------
    # Task 3: Model Training
    # --------------------------------------------------------
    def model_trainer(**kwargs):
        """
        This task trains the machine learning model using preprocessed data.
        """
        import numpy as np
        ti = kwargs['ti']
        data_transformation_artifact = ti.xcom_pull(
            task_ids='data_transformation',
            key='data_transformation_artifact'
        )
        train_arr = np.array(data_transformation_artifact['train_arr'])
        test_arr = np.array(data_transformation_artifact['test_arr'])
        training_pipeline.start_training(train_arr, test_arr)

    # --------------------------------------------------------
    # Task 4: Push Artifacts to S3
    # --------------------------------------------------------
    def push_data_to_s3(**kwargs):
        """
        This task uploads model artifacts and outputs to an S3 bucket.
        """
        import os
        bucket_name = os.getenv('BUCKET_NAME')  # Read bucket name from environment variable
        artifact_folder = '/app/artifacts'
        # Upload folder contents to S3
        os.system(f'aws s3 sync {artifact_folder} s3://{bucket_name}/artifact')

    # --------------------------------------------------------
    # Airflow Task Definitions
    # --------------------------------------------------------
    data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent("""
        ### Data Ingestion Task
        This task reads raw data, performs train-test split,
        and saves dataset files for transformation.
    """)

    data_transform_task = PythonOperator(
        task_id='data_transformation',
        python_callable=data_transformation,
    )
    data_transform_task.doc_md = dedent("""
        ### Data Transformation Task
        This task performs feature engineering and preprocessing.
    """)

    model_trainer_task = PythonOperator(
        task_id='model_trainer',
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent("""
        ### Model Training Task
        This task trains the ML model using prepared datasets.
    """)

    push_data_to_s3_task = PythonOperator(
        task_id='push_data_to_s3',
        python_callable=push_data_to_s3,
    )
    push_data_to_s3_task.doc_md = dedent("""
        ### S3 Upload Task
        This task uploads model artifacts and logs to AWS S3 for storage.
    """)

    # --------------------------------------------------------
    # Task Dependency Chain (Flow of Execution)
    # --------------------------------------------------------
    data_ingestion_task >> data_transform_task >> model_trainer_task >> push_data_to_s3_task
