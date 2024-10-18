# model training
from pyparsing import Any, Union
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from modules.dataPreprocessing.strategy import Strategy


class ModelTrainer:
    def __init__(
        self,
        Estimator: Union[DecisionTreeClassifier, RandomForestClassifier],
        Features: pd.DataFrame,
        Target: pd.DataFrame,
        Strategy: Strategy,
    ):
        self.Strategy = Strategy
        self.Estimator = Estimator
        self.Features = Features
        self.Target = Target

    def getStrategy(self) -> None:
        return self.Strategy

    def setStrategy(self, Strategy: Strategy) -> None:
        self.Strategy = Strategy

    def fitModel(self) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
        return self.Strategy.fitModel(self, self.Estimator, self.Features, self.Target)
