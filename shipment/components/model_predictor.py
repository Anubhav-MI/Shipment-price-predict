import sys
import pickle
import pandas as pd
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

    def get_data(self) -> dict:
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
            df = pd.DataFrame(input_dict)
            logging.info("Exited get_input_data_frame method of shippingData class")
            return df
        except Exception as e:
            raise ShipmentException(e, sys)




class CostPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            self.model_path = os.path.join(
                os.getcwd(), "artifacts", "ModelTrainerArtifacts", "shipping_price_model.pkl"
            )
        else:
            self.model_path = model_path

    def load_model(self):
        try:
            logging.info(f"Loading model from: {self.model_path}")
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            raise ShipmentException(e, sys)
        
        

    def predict(self, X: pd.DataFrame) -> list[float]:
        """
        Returns a list of predicted costs for the rows in X.
        """
        logging.info("Entered predict method of CostPredictor class")
        try:
            model = self.load_model()
            predictions = model.predict(X)
            logging.info("Exited predict method of CostPredictor class")
            # ensure Python floats
            return [float(p) for p in predictions]
        except Exception as e:
            raise ShipmentException(e, sys)
