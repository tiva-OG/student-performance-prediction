import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            scaled_data = preprocessor.transform(features)
            predictions = model.predict(scaled_data)

            return predictions
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        writing_score=72,
        reading_score=72
    )
    df = data.get_data_as_dataframe()
    print(df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(df)
    print(f"RESULTS: {results}")
