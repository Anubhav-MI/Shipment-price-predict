import sys
import os
import pandas as pd
from shipment.components.data_ingestion import DataIngestion
from shipment.components.data_validation import DataValidation
from shipment.components.data_transformation import DataTransformation
from shipment.components.model_trainer import ModelTrainer
from shipment.components.model_predictor import CostPredictor, shippingData

from shipment.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)

from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging

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
        if data_validation_artifact.validation_status:
            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            logging.info("Initiating data transformation...")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed.")
            print(data_transformation_artifact)

            # 4. Model Training (save model even if base score not met)
            model_trainer_config = ModelTrainerConfig(training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
            logging.info("Initiating model training...")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed.")
            print(model_trainer_artifact)

            # 5. Prediction (console only)
            model_path = model_trainer_artifact.trained_model_file_path

            if os.path.exists(model_path):
                sample_input = shippingData(
                    artistReputation=5,
                    height=12,
                    width=10,
                    weight=8,
                    material="Marble",
                    priceOfSculpture=20000,
                    baseShippingPrice=500,
                    international="Yes",
                    expressShipment="No",
                    installationIncluded="Yes",
                    transport="Airways",
                    fragile="Yes",
                    customerInformation="Wealthy",
                    remoteLocation="No"
                )
                input_df = sample_input.get_input_data_frame()
                predictor = CostPredictor(model_path=model_path)
                model = predictor.load_model()
                prediction = model.predict(input_df)[0]
                cost_value = round(float(prediction), 2)
                print(f"\n✅ Sample shipment cost prediction: {cost_value}")
            else:
                print(f"\n❌ Model file not found at: {model_path}. Cannot make prediction.")

        else:
            logging.warning("Skipping data transformation, model training, and prediction due to failed validation.")
            print("⚠️ Skipping data transformation, model training, and prediction due to failed validation.")

    except Exception as e:
        raise ShipmentException(e, sys)
