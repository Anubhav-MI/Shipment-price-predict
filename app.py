from shipment.components.data_ingestion import DataIngestion
from shipment.components.data_validation import DataValidation
from shipment.exception.exception import ShipmentException
from shipment.logging.logger import logging
from shipment.entity.config_entity import DataIngestionConfig,DataValidationConfig
from shipment.entity.config_entity import TrainingPipelineConfig

 

import sys

if __name__ =='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig (trainingpipelineconfig)
        data_ingestion=DataIngestion (dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=data_validation.initiate_data_validation() 
        logging.info("Data Vlidation Completed")
        print(data_validation_artifact)
        
        
    except Exception as e:
           raise ShipmentException(e,sys)