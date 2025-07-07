import os
import sys
import yaml
import pickle
import numpy as np
import dill
import shutil
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging
from shipment.constant.training_pipeline import MODEL_CONFIG_FILE
from typing import Tuple


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ShipmentException(e, sys) from e


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ShipmentException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ShipmentException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ShipmentException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ShipmentException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ShipmentException(e, sys) from e


class MainUtils:
    def read_yaml_file(self, file_path: str) -> dict:
        return read_yaml_file(file_path)

    def save_numpy_array_data(self, file_path: str, array: np.array):
        return save_numpy_array_data(file_path, array)

    def load_numpy_array_data(self, file_path: str) -> np.array:
        return load_numpy_array_data(file_path)

    def save_object(self, file_path: str, obj: object):
        return save_object(file_path, obj)

    def load_object(self, file_path: str) -> object:
        return load_object(file_path)

    def get_tuned_model(self, model_name, train_x, train_y, test_x, test_y):
        logging.info("Entered the get_tuned_model method of MainUtils class")
        try:
            model = self.get_base_model(model_name)
            model_best_params = self.get_model_params(model, train_x, train_y)
            model.set_params(**model_best_params)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            model_score = self.get_model_score(test_y, preds)
            logging.info("Exited the get_tuned_model method of MainUtils class")
            return model_score, model, model.__class__.__name__
        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def get_model_score(test_y, preds):
        logging.info("Entered the get_model_score method of MainUtils class")
        try:
            score = r2_score(test_y, preds)
            logging.info(f"Model score is {score}")
            logging.info("Exited the get_model_score method of MainUtils class")
            return score
        except Exception as e:
            raise ShipmentException(e, sys) from e

    @staticmethod
    def get_base_model(model_name: str) -> object:
        logging.info("Entered the get_base_model method of MainUtils class")
        try:
            if model_name.lower().startswith("xgb"):
                model = xgboost.__dict__[model_name]()
            else:
                model_idx = [model[0] for model in all_estimators()].index(model_name)
                model = all_estimators()[model_idx][1]()
            logging.info("Exited the get_base_model method of MainUtils class")
            return model
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_model_params(self, model, x_train, y_train):
        logging.info("Entered the get_model_params method of MainUtils class")
        try:
            model_name = model.__class__.__name__
            model_config = self.read_yaml_file(MODEL_CONFIG_FILE)
            param_grid = model_config["train_model"][model_name]

            grid = GridSearchCV(model, param_grid, cv=2, n_jobs=-1, verbose=3)
            grid.fit(x_train, y_train)

            logging.info("Exited the get_model_params method of MainUtils class")
            return grid.best_params_
        except Exception as e:
            raise ShipmentException(e, sys) from e
    @staticmethod
    def get_best_model_with_name_and_score(model_list: list) -> Tuple[object, float]:
        logging.info("Entered the get_best_model_with_name_and_score method of MainUtils class")
        try:
            # model_list: List of tuples like (score, model)
            best_score = max(model_list, key=lambda x: x[0])[0]
            best_model = max(model_list, key=lambda x: x[0])[1]
            logging.info("Exited the get_best_model_with_name_and_score method of MainUtils class")
            return best_model, best_score
        except Exception as e:
            raise ShipmentException(e, sys) from e
