from shipment.components.data_ingestion import DataIngestion
from shipment.components.data_validation import DataValidation
from shipment.components.data_transformation import DataTransformation
from model_trainer.main import ModelTrainer
from shipment.components.model_evaluation import ModelEvaluation

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()

    def run_pipeline(self):
        # Step 1: Data Ingestion
        ingestion_artifact = self.data_ingestion.initiate_data_ingestion()

        # Step 2: Data Validation
        validation_artifact = self.data_validation.initiate_data_validation(ingestion_artifact)

        # Step 3: Data Transformation
        transformation_artifact = self.data_transformation.initiate_data_transformation(validation_artifact)

        # Step 4: Model Training
        trainer_artifact = self.model_trainer.initiate_model_trainer(transformation_artifact)

        # Step 5: Model Evaluation
        evaluation_artifact = self.model_evaluation.initiate_model_evaluation(trainer_artifact, transformation_artifact)

        return "Training pipeline completed successfully!"