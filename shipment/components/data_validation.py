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
    
    def _preprocess_for_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataframe for drift detection"""
        df_copy = df.copy()
        
        # Encode categorical columns
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
            
            # Preprocess dataframes
            base_df_processed = self._preprocess_for_drift(base_df)
            current_df_processed = self._preprocess_for_drift(current_df)
            
            # Check for all expected columns
            all_columns = set(base_df_processed.columns).union(set(current_df_processed.columns))
            
            for column in all_columns:
                if column not in base_df_processed.columns or column not in current_df_processed.columns:
                    report.update({column: {
                        "p_value": None,
                        "drift_status": "Column missing in one of the datasets"
                    }})
                    status = False
                    continue
                    
                d1 = base_df_processed[column]
                d2 = current_df_processed[column]
                
                # Handle missing values
                d1 = d1.replace([np.inf, -np.inf], np.nan).dropna()
                d2 = d2.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(d1) == 0 or len(d2) == 0:
                    report.update({column: {
                        "p_value": None,
                        "drift_status": "Insufficient data after cleaning"
                    }})
                    status = False
                    continue
                
                # Perform KS test
                is_same_dist = ks_2samp(d1, d2)
                
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            
            # Save drift report
            dir_path = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report)
            
            return status, report
            
        except Exception as e:
            raise ShipmentException(e, sys)
    
    def validate_data_types(self, dataframe: pd.DataFrame) -> bool:
        """Validate that columns have correct data types"""
        try:
            status = True
            type_report = {}
            
            for col in self.numerical_columns:
                if col in dataframe.columns:
                    if not np.issubdtype(dataframe[col].dtype, np.number):
                        type_report[col] = {
                            "expected": "numeric",
                            "actual": str(dataframe[col].dtype),
                            "status": False
                        }
                        status = False
                    else:
                        type_report[col] = {
                            "expected": "numeric",
                            "actual": str(dataframe[col].dtype),
                            "status": True
                        }
            
            for col in self.categorical_columns:
                if col in dataframe.columns:
                    type_report[col] = {
                        "expected": "categorical",
                        "actual": str(dataframe[col].dtype),
                        "status": True  # We'll accept any type for categorical
                    }
            
            # Save type validation report
            dir_path = os.path.dirname(self.data_validation_config.data_type_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=self.data_validation_config.data_type_report_file_path, 
                          content=type_report)
            
            return status
        except Exception as e:
            raise ShipmentException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            # Validate number of columns
            train_col_status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not train_col_status:
                error_message += "Train dataframe does not contain all required columns.\n"
            
            test_col_status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not test_col_status:
                error_message += "Test dataframe does not contain all required columns.\n"   
            
            # Validate data types
            train_type_status = self.validate_data_types(train_dataframe)
            if not train_type_status:
                error_message += "Train dataframe has incorrect data types.\n"
            
            test_type_status = self.validate_data_types(test_dataframe)
            if not test_type_status:
                error_message += "Test dataframe has incorrect data types.\n"
            
            # Check for data drift
            drift_status, _ = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )
            
            # Prepare directories and save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, 
                index=False, 
                header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, 
                index=False, 
                header=True
            )
            
            # Create validation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status and train_col_status and test_col_status 
                               and train_type_status and test_type_status,
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