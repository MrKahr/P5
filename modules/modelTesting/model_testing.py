from typing import Callable

from sklearn import metrics
from modules.config.config_enums import ModelScoreFunc
from modules.logging import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from modules.config.config import Config
from modules.types import FittedEstimator


class ModelTester:
    def __init__(
        self,
        estimator: FittedEstimator,
        train_x: pd.DataFrame,
        target_y: pd.Series,
        model_report: dict,
    ):
        self._config = Config()
        self._estimator = estimator
        self._train_x = train_x
        self._target_y = target_y
        self._model_report = model_report

    def run(self) -> None:
        train_features, test_features = train_test_split(features)
        train_target, test_target = train_test_split(target)
        prediction = estimator.predict(test_features)
        tn, fp, fn, tp = confusion_matrix(test_target, prediction)
        accuracy = accuracy_score(test_target, prediction)
        precision = precision_score(test_target, prediction)
        recall = recall_score(test_target, prediction)
        specificity = tn / (tn + fn)
        score = self.custom_score(estimator, test_features, test_target)

        print(
            f"Custom scoring on training data: {self.custom_score(estimator, train_features, train_target):.3f}"
        )
        print(
            f"Custom scoring on test data: {self.custom_score(estimator, test_features, test_target):.3f}"
        )

        self._model_report = {
            "feature_importances": None,  # type: Bunch
            "feature_names_in": None,  # type: NDArray
            "feature_count": None,  # type: int
            "custom_scoring": score,  # type: float
            "true_positive": tp,  # type: float
            "true_negative": tn,  # type: float
            "false_positive": fp,  # type: float
            "false_negative": fn,  # type: float
            "accuracy": accuracy,  # type: float
            "precision": precision,  # type: float
            "recall": recall,  # type: float
            "specificity": specificity,  # type: float
        }

        logger.info(f"ModelTester is done")
