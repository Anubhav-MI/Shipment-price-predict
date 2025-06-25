from shipment.components.data_ingestion import DataIngestion
from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging
from shipment.entity.config_entity import DataIngestionConfig
from shipment.entity.config_entity import TrainingPipelineConfig

 

import sys

if __name__ =='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig (trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
    
        
        
        
    except Exception as e:
           raise ShipmentException(e,sys)