import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders.binary import BinaryEncoder

from shipment.logging.logger import logging
from shipment.utils.main_utils.utils import MainUtils
from shipment.constant.training_pipeline import SCHEMA_FILE_PATH
from shipment.exception.exception import ShipmentException
from shipment.entity.config_entity import DataTransformationConfig
from shipment.entity.artifacts_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifacts: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_validation_artifacts = data_validation_artifacts
        self.data_transformation_config = data_transformation_config

        self.UTILS = MainUtils()
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(SCHEMA_FILE_PATH)

        self.train_set = pd.read_csv(self.data_validation_artifacts.valid_train_file_path)
        self.test_set = pd.read_csv(self.data_validation_artifacts.valid_test_file_path)

    def get_data_transformer_object(self) -> object:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numerical_columns = self.SCHEMA_CONFIG["numerical_columns"]
            onehot_columns = self.SCHEMA_CONFIG["onehot_columns"]
            binary_columns = self.SCHEMA_CONFIG["binary_columns"]

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            binary_transformer = BinaryEncoder()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_transformer, onehot_columns),
                    ("BinaryEncoder", binary_transformer, binary_columns),
                    ("StandardScaler", numeric_transformer, numerical_columns),
                ]
            )

            logging.info("Created preprocessor object")
            return preprocessor

        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def _outlier_capping(col, df: DataFrame) -> DataFrame:
        logging.info(f"Outlier capping for column: {col}")
        try:
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr

            df.loc[df[col] > upper_limit, col] = upper_limit
            df.loc[df[col] < lower_limit, col] = lower_limit

            return df

        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation process")

        try:
            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            logging.info(f"Created transformation dir at {self.data_transformation_config.data_transformation_dir}")

            preprocessor = self.get_data_transformer_object()

            target_column_name = self.SCHEMA_CONFIG["target_column"]
            numerical_columns = self.SCHEMA_CONFIG["numerical_columns"]

            continuous_columns = [col for col in numerical_columns if len(self.train_set[col].unique()) >= 25]

            for col in continuous_columns:
                self._outlier_capping(col, self.train_set)
                self._outlier_capping(col, self.test_set)

            input_feature_train_df = self.train_set.drop(columns=[target_column_name])
            target_feature_train_df = self.train_set[target_column_name]

            input_feature_test_df = self.test_set.drop(columns=[target_column_name])
            target_feature_test_df = self.test_set[target_column_name]

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            train_dir = os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            test_dir = os.path.dirname(self.data_transformation_config.transformed_test_file_path)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            transformed_train_file = self.UTILS.save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, train_arr
            )
            transformed_test_file = self.UTILS.save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, test_arr
            )

            preprocessor_obj_file = self.UTILS.save_object(
                self.data_transformation_config.preprocessor_object_file_path, preprocessor
            )

            logging.info("Completed data transformation")

            return DataTransformationArtifact(
                transformed_object_file_path=preprocessor_obj_file,
                transformed_train_file_path=transformed_train_file,
                transformed_test_file_path=transformed_test_file,
            )

        except Exception as e:
            raise ShipmentException(e, sys) from e
