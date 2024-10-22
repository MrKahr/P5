from modules.logging import logger
import pandas as pd
from sklearn.model_selection import train_test_split


class ModelTester:
    def __init__(self):
        pass

    def custom_score(estimator, X, y):
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

    @classmethod
    def run(self, features: pd.DataFrame, target: pd.DataFrame, estimator) -> None:

        train_features, test_features = train_test_split(features, 0.8)
        train_target, test_target = train_test_split(target)

        print(
            f"Training data accuracy: {self.custom_score(estimator, train_features, train_target):.3f}"
        )
        print(
            f"Testing data accuracy: {self.custom_score(estimator, test_features, test_target):.3f}"
        )
        logger.info(f"ModelTester is done")
