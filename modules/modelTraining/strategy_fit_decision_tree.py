from typing import Any

from sklearn.tree import DecisionTreeClassifier
from modules.dataPreprocessing.strategy import Strategy


class StrategyFitDecisionTree:
    def fitModel(self, estimator: DecisionTreeClassifier, features, target) -> Any:
        return estimator.fit(features, target)
