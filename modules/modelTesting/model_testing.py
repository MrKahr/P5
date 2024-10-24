from typing import Callable

from sklearn import metrics
from modules.config.config_enums import ScoreFunction
from modules.logging import logger
import pandas as pd
from sklearn.model_selection import train_test_split
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

        train_features, test_features = train_test_split(features, 0.8)
        train_target, test_target = train_test_split(target)

        print(
            f"Training data accuracy: {self.customScore(estimator, train_features, train_target):.3f}"
        )
        print(
            f"Testing data accuracy: {self.customScore(estimator, test_features, test_target):.3f}"
        )
        result = estimator.predict(X)

        logger.info(f"ModelTester is done")
