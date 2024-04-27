import os
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.num_columns = ["writing_score", "reading_score"]
        self.cat_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ]

    def get_preprocessor(self):
        """
        function responsible for initialising the data transformation object
        """
        try:

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns: {self.cat_columns}")
            logging.info(f"Numerical Columns: {self.num_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, self.num_columns),
                    ("cat_pipeline", cat_pipeline, self.cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info("Initiating data transformation")

        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Read the train and test data as pandas dataframe")

            logging.info("Obtaining the preprocessor object")
            preprocessor = self.get_preprocessor()

            target_column = "math_score"

            X_train = train_data.drop(columns=[target_column], axis=1)
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column], axis=1)
            y_test = test_data[target_column]

            logging.info(f"Applying transformation on train and test data")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("Transformation completed. Saving preprocessor object")
            save_object(
                obj=preprocessor,
                file_path=self.transformation_config.preprocessor_path
            )

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e, sys)
