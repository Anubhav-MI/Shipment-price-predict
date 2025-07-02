import os
import sys
import pandas as pd
from typing import List, Tuple
from pandas import DataFrame

from shipment.logging.logger import logging
from shipment.constant.training_pipeline import MODEL_CONFIG_FILE
from shipment.entity.config_entity import ModelTrainerConfig
from shipment.entity.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact
from shipment.exception.exception import ShipmentException


class CostModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X) -> float:
        logging.info("Entered predict method of CostModel class")
        try:
            transformed_feature = self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_trained_models(
        self, x_data: DataFrame, y_data: DataFrame
    ) -> List[Tuple[float, object, str]]:
        logging.info("Entered get_trained_models method of ModelTrainer class")
        try:
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                MODEL_CONFIG_FILE
            )
            models_list = list(model_config["train_model"].keys())
            logging.info(f"Got model list from config: {models_list}")

            x_train = x_data.iloc[:, :-1]
            y_train = x_data.iloc[:, -1]
            x_test = y_data.iloc[:, :-1]
            y_test = y_data.iloc[:, -1]

            tuned_model_list = [
                self.model_trainer_config.UTILS.get_tuned_model(
                    model_name, x_train, y_train, x_test, y_test
                )
                for model_name in models_list
            ]

            return tuned_model_list
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            logging.info(f"Created model trainer artifacts directory at: {self.model_trainer_config.model_trainer_dir}")

            # Load train and test arrays
            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            train_df = pd.DataFrame(train_array)

            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            test_df = pd.DataFrame(test_array)

            # Get best model from training
            list_of_trained_models = self.get_trained_models(train_df, test_df)
            best_model, best_model_score = (
                self.model_trainer_config.UTILS.get_best_model_with_name_and_score(
                    list_of_trained_models
                )
            )

            # Load preprocessing object
            preprocessing_obj = self.model_trainer_config.UTILS.load_object(
                self.model_trainer_config.preprocessor_object_file_path
            )

            # Load base model score from config
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                MODEL_CONFIG_FILE
            )
            base_model_score = float(model_config["base_model_score"])

            if best_model_score >= base_model_score:
                cost_model = CostModel(preprocessing_obj, best_model)
                trained_model_path = self.model_trainer_config.trained_model_file_path
                self.model_trainer_config.UTILS.save_object(
                    trained_model_path, cost_model
                )
                logging.info("Saved best cost model")
            else:
                logging.warning("No model surpassed base model score")
                raise Exception("No best model found with score more than base score")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path
            )

        except Exception as e:
            raise ShipmentException(e, sys) from e
