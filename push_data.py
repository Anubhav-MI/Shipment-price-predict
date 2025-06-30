import os
import sys
import json
from dotenv import load_dotenv 

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()
 
import pandas as pd
import numpy as np
import pymongo
from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging

class ShipmentDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ShipmentException(e,sys)
        
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = json.loads(data.to_json(orient='records'))
            return records
        except Exception as e:
            raise ShipmentException(e, sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database= database
            self.collection = collection
            self.records = records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise ShipmentException(e,sys)
        
if __name__ == '__main__':
    FIEL_PATH = "shipment_data\data\shipment.csv"
    DATABASE = "SHIPMENT"
    Collection = "ShipmentData"
    shipmentobj = ShipmentDataExtract()
    records = shipmentobj.csv_to_json_converter(file_path=FIEL_PATH)
    print(records)
    no_of_records = shipmentobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
