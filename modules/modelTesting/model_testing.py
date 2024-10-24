from typing import Callable

from sklearn import metrics
from modules.config.config_enums import ScoreFunction
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


class ModelTester:
    def __init__(self, config: Config):
        self.scoreFunction = self.scoreFunctionSelector(config.getValue("ScoreFunc"))

    def customScore(estimator, X, y):
        def threshold(result, y, th: int = 5):
            score = 0
            for i, prediction in enumerate(result):
                if y[i] >= th:
                    score += 1
                elif prediction < th:
                    score += 1
            return score

        result = estimator.predict(X)
        score = threshold(result, y)
        # score = threshold(result, y, th=20)
        return float(score / len(result))

    def scoreFunctionSelector(self, scoreFunction: ScoreFunction) -> Callable | None:
        currentScoreFunction = None
        match scoreFunction:
            case ScoreFunction.CUSTOM_SCORE_FUNC:
                currentScoreFunction = self.customScore
            case ScoreFunction.ACCURACY:
                currentScoreFunction = metrics.accuracy_score
            case ScoreFunction.BALANCED_ACCURACY:
                currentScoreFunction = metrics.balanced_accuracy_score
            case ScoreFunction.EXPLAINED_VARIANCE:
                currentScoreFunction = metrics.explained_variance_score
        return currentScoreFunction

    def getScoreFunc(self) -> None:
        return self.scoreFunction

    def run(self, features: pd.DataFrame, target: pd.DataFrame, estimator) -> None:

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
