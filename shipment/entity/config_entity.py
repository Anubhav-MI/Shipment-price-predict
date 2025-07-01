from datetime import datetime
import os

from shipment.constant import training_pipeline
from shipment.utils.main_utils.utils import MainUtils  
from shipment.constant.training_pipeline import SCHEMA_FILE_PATH
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp

         
class DataIngestionConfig:
    def __init__(self,training_pipeline_config):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
            )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME




# @dataclass
# class DataValidationConfig:
#     def _init_(self):
#         self.UTILS = MainUtils()
#         self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
#         self.DATA_INGESTION_ARTIFCATS_DIR: str = os.path.join(
#             from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR
#         )
#         self.DATA_VALIDATION_ARTIFACTS_DIR: str = os.path.join(
#             from_root(), ARTIFACTS_DIR, DATA_VALIDATION_ARTIFACT_DIR
#         )
#         self.DATA_DRIFT_FILE_PATH: str = os.path.join(
#             self.DATA_VALIDATION_ARTIFACTS_DIR, DATA_DRIFT_FILE_NAME
#         )



class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_FILE_NAME
        )

        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )

        # ✅ ADD THIS MISSING LINE
        self.data_type_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DATA_TYPE_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DATA_TYPE_REPORT_FILE_NAME,
        )




class DataTransformationConfig:
    def __init__(self, training_pipeline_config):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # Path to transformed data
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRANSFORMED_TRAIN_FILE_NAME
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRANSFORMED_TEST_FILE_NAME
        )

        # Path to preprocessor (like StandardScaler, ColumnTransformer, etc.)
        self.preprocessor_object_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_PREPROCESSOR_DIR,
            training_pipeline.PREPROCESSOR_OBJECT_FILE_NAME
        )

class ModelTrainerConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFCATS_DIR
        )
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(
            from_root(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR
        )
        self.PREPROCESSOR_OBJECT_FILE_PATH: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR, PREPROCESSOR_OBJECT_FILE_NAME
        )
        self.TRAINED_MODEL_FILE_PATH: str = os.path.join(
            from_root(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR, MODEL_FILE_NAME
        )

        # # Utility and Schema
        # self.UTILS = MainUtils()
        # self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(SCHEMA_FILE_PATH)

        # # For saving in initiate_data_transformation()
        # self.TRANSFORMED_TRAIN_DATA_DIR = os.path.dirname(self.transformed_train_file_path)
        # self.TRANSFORMED_TEST_DATA_DIR = os.path.dirname(self.transformed_test_file_path)
        # self.PREPROCESSOR_FILE_PATH = self.preprocessor_object_file_path

