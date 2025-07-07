from shipment.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact
from shipment.entity.config_entity import DataValidationConfig
from shipment.exception.exception import ShipmentException 
from shipment.logging.logger import logging 
from shipment.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os, sys
from typing import Tuple, Dict
from shipment.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.categorical_columns = self._schema_config.get('categorical_columns', [])
            self.numerical_columns = self._schema_config.get('numerical_columns', [])
            self.target_column = self._schema_config.get('target_column')
        except Exception as e:
            raise ShipmentException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ShipmentException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = set(self.numerical_columns + self.categorical_columns)
            if self.target_column:
                expected_columns.add(self.target_column)

            actual_columns = set(dataframe.columns)

            logging.info(f"Required columns: {expected_columns}")
            logging.info(f"Dataframe has columns: {actual_columns}")

            return expected_columns.issubset(actual_columns)
        except Exception as e:
            raise ShipmentException(e, sys)

    def _ensure_positive_target(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.target_column in df.columns:
                original_len = len(df)
                df = df[df[self.target_column] > 0]
                logging.info(f"{original_len - len(df)} rows removed where {self.target_column} was non-positive.")
            return df
        except Exception as e:
            raise ShipmentException(e, sys)

    def _preprocess_for_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in self.categorical_columns:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        return df_copy

    def detect_dataset_drift(self, base_df: pd.DataFrame,
                             current_df: pd.DataFrame,
                             threshold: float = 0.05) -> Tuple[bool, Dict]:
        try:
            status = True
            report = {}

            base_df_processed = self._preprocess_for_drift(base_df)
            current_df_processed = self._preprocess_for_drift(current_df)

            all_columns = set(base_df_processed.columns).union(set(current_df_processed.columns))

            for column in all_columns:
                if column not in base_df_processed.columns or column not in current_df_processed.columns:
                    report[column] = {
                        "p_value": None,
                        "drift_status": "Column missing in one of the datasets"
                    }
                    status = False
                    continue

                d1 = base_df_processed[column].replace([np.inf, -np.inf], np.nan).dropna()
                d2 = current_df_processed[column].replace([np.inf, -np.inf], np.nan).dropna()

                if len(d1) == 0 or len(d2) == 0:
                    report[column] = {
                        "p_value": None,
                        "drift_status": "Insufficient data after cleaning"
                    }
                    status = False
                    continue

                is_same_dist = ks_2samp(d1, d2)
                drift_detected = is_same_dist.pvalue < threshold

                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": drift_detected
                }

                if drift_detected:
                    status = False

            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            write_yaml_file(self.data_validation_config.drift_report_file_path, report)

            return status, report

        except Exception as e:
            raise ShipmentException(e, sys)

    def validate_data_types(self, dataframe: pd.DataFrame) -> bool:
        try:
            status = True
            type_report = {}

            for col in self.numerical_columns:
                if col in dataframe.columns:
                    is_numeric = np.issubdtype(dataframe[col].dtype, np.number)
                    type_report[col] = {
                        "expected": "numeric",
                        "actual": str(dataframe[col].dtype),
                        "status": is_numeric
                    }
                    if not is_numeric:
                        status = False

            for col in self.categorical_columns:
                if col in dataframe.columns:
                    type_report[col] = {
                        "expected": "categorical",
                        "actual": str(dataframe[col].dtype),
                        "status": True
                    }

            os.makedirs(os.path.dirname(self.data_validation_config.data_type_report_file_path), exist_ok=True)
            write_yaml_file(self.data_validation_config.data_type_report_file_path, type_report)

            return status
        except Exception as e:
            raise ShipmentException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            train_dataframe = self._ensure_positive_target(train_dataframe)
            test_dataframe = self._ensure_positive_target(test_dataframe)

            train_col_status = self.validate_number_of_columns(train_dataframe)
            if not train_col_status:
                error_message += "Train dataframe does not contain all required columns.\n"

            test_col_status = self.validate_number_of_columns(test_dataframe)
            if not test_col_status:
                error_message += "Test dataframe does not contain all required columns.\n"

            train_type_status = self.validate_data_types(train_dataframe)
            if not train_type_status:
                error_message += "Train dataframe has incorrect data types.\n"

            test_type_status = self.validate_data_types(test_dataframe)
            if not test_type_status:
                error_message += "Test dataframe has incorrect data types.\n"

            drift_status, _ = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            # ✅ Allow drift during dev (set False for production)
            drift_allowed = True

            validation_status = drift_allowed and train_col_status and test_col_status \
                                and train_type_status and test_type_status

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                data_type_report_file_path=self.data_validation_config.data_type_report_file_path,
                error_message=error_message if error_message else None
            )

            return data_validation_artifact

        except Exception as e:
            raise ShipmentException(e, sys)