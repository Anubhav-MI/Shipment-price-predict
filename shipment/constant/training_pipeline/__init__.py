from datetime import datetime
import os
from shipment.constant import training_pipeline

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "Cost"
PIPELINE_NAME: str = "Shipment"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "shipment.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

MODEL_CONFIG_FILE = "data_schema/model.yaml"
SCHEMA_FILE_PATH = "data_schema/schema.yaml"


# SAVED_MODEL_DIR =os.path.join("saved_models")
# MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "ShipmentData"
DATA_INGESTION_DATABASE_NAME: str = "SHIPMENT"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

# DATA_VALIDATION_ARTIFACT_DIR = "DataValidationArtifacts"
# DATA_DRIFT_FILE_NAME = "DataDriftReport.yaml"

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DATA_TYPE_REPORT_DIR: str = "data_type_report"
DATA_VALIDATION_DATA_TYPE_REPORT_FILE_NAME: str = "data_type_report.yaml"

DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
TRANSFORMED_TRAIN_FILE_NAME: str = "transformed_train_data.npz"
TRANSFORMED_TEST_FILE_NAME: str = "transformed_test_data.npz"

DATA_TRANSFORMATION_PREPROCESSOR_DIR: str = "preprocessor"
PREPROCESSOR_OBJECT_FILE_NAME: str = "shipping_preprocessor.pkl"


MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
MODEL_FILE_NAME = "shipping_price_model.pkl"
MODEL_SAVE_FORMAT = ".pkl"
