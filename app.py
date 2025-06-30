from shipment.components.data_ingestion import DataIngestion
from shipment.components.data_validation import DataValidation
from shipment.components.data_transformation import DataTransformation

from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging

from shipment.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    TrainingPipelineConfig,
)

import sys

if __name__ == '__main__':
    try:
        # Initialize training pipeline config
        training_pipeline_config = TrainingPipelineConfig()

        # 1. Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")
        print(data_ingestion_artifact)

        # 2. Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiating data validation...")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed.")
        print(data_validation_artifact)

        # 3. Data Transformation
        if data_validation_artifact.validation_status:  # Run transformation only if validation is successful
            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            logging.info("Initiating data transformation...")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed.")
            print(data_transformation_artifact)
        else:
            logging.warning("Skipping data transformation due to failed data validation.")
            print("⚠️ Skipping data transformation due to failed validation.")

    except Exception as e:
        raise ShipmentException(e, sys)
