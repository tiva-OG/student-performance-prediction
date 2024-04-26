import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)  # train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
