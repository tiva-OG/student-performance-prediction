import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def train(self):
        try:
            logging.info("Model training just begun")
            # ingesting data to system
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            # transforming data for training
            train_array, test_array = self.data_transformation.initiate_data_transformation(train_data_path,
                                                                                            test_data_path)
            # initiating training w/ model-trainer
            best_model_score = self.model_trainer.initiate_model_trainer(train_array, test_array)
            logging.info(f"Best model score: {best_model_score}")
            logging.info("Model training completed.")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.train()
