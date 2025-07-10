# shipment/pipeline/train_pipeline.py

import os
import sys
from shipment.components.data_ingestion import DataIngestion
from shipment.components.data_validation import DataValidation
from shipment.components.data_transformation import DataTransformation
from shipment.components.model_trainer import ModelTrainer
from shipment.components.model_predictor import CostPredictor, shippingData
from shipment.components.model_evaluation import ModelEvaluation

from shipment.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)

from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging


def run_training_pipeline():
    try:
        training_pipeline_config = TrainingPipelineConfig()

        # 1. Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("🚚 Starting data ingestion...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # 2. Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("🔍 Starting data validation...")
        data_validation_artifact = data_validation.initiate_data_validation()

        # 3. Data Transformation
        if data_validation_artifact.validation_status:
            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            # 4. Model Training
            model_trainer_config = ModelTrainerConfig(training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            # 5. Sample Prediction
            model_path = model_trainer_artifact.trained_model_file_path
            cost_value = None
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
                prediction = predictor.predict(input_df)[0]
                cost_value = round(float(prediction), 2)

            # 6. Model Evaluation
            model_eval = ModelEvaluation(
                model_path=model_path,
                test_data_path=data_validation_artifact.valid_test_file_path
            )
            evaluation_result = model_eval.evaluate()

            return {
                "status": "Pipeline executed successfully ✅",
                # "prediction_sample_cost": cost_value,
                # "evaluation_metrics": evaluation_result
            }
        else:
            msg = "⚠️ Validation failed. Skipping pipeline."
            logging.warning(msg)
            return {"status": msg}

    except Exception as e:
        raise ShipmentException(e, sys)
