import sys
import pickle
import pandas as pd
from typing import Dict
from shipment.logging.logger import logging
from shipment.exception.exception import ShipmentException
import os


class shippingData:
    def __init__(
        self,
        artistReputation,
        height,
        width,
        weight,
        material,
        priceOfSculpture,
        baseShippingPrice,
        international,
        expressShipment,
        installationIncluded,
        transport,
        fragile,
        customerInformation,
        remoteLocation,
    ):
        self.artistReputation = artistReputation
        self.height = height
        self.width = width
        self.weight = weight
        self.material = material
        self.priceOfSculpture = priceOfSculpture
        self.baseShippingPrice = baseShippingPrice
        self.international = international
        self.expressShipment = expressShipment
        self.installationIncluded = installationIncluded
        self.transport = transport
        self.fragile = fragile
        self.customerInformation = customerInformation
        self.remoteLocation = remoteLocation

    def get_data(self) -> Dict:
        logging.info("Entered get_data method of shippingData class")
        try:
            input_data = {
                "Artist Reputation": [self.artistReputation],
                "Height": [self.height],
                "Width": [self.width],
                "Weight": [self.weight],
                "Material": [self.material],
                "Price Of Sculpture": [self.priceOfSculpture],
                "Base Shipping Price": [self.baseShippingPrice],
                "International": [self.international],
                "Express Shipment": [self.expressShipment],
                "Installation Included": [self.installationIncluded],
                "Transport": [self.transport],
                "Fragile": [self.fragile],
                "Customer Information": [self.customerInformation],
                "Remote Location": [self.remoteLocation],
            }
            logging.info("Exited get_data method of shippingData class")
            return input_data
        except Exception as e:
            raise ShipmentException(e, sys)

    def get_input_data_frame(self) -> pd.DataFrame:
        logging.info("Entered get_input_data_frame method of shippingData class")
        try:
            input_dict = self.get_data()
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise ShipmentException(e, sys)


class CostPredictor:
    def __init__(self, model_path: str = os.path.join("saved_models", "best_model_tuned_DecisionTree.pkl")):
        self.model_path = model_path

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise ShipmentException(e, sys)

    def predict(self, X: pd.DataFrame) -> float:
        logging.info("Entered predict method of CostPredictor class")
        try:
            model = self.load_model()
            prediction = model.predict(X)
            logging.info("Exited predict method of CostPredictor class")
            return prediction
        except Exception as e:
            raise ShipmentException(e, sys)


