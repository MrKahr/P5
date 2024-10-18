# model training
from pyparsing import Any, Union
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from modules.dataPreprocessing.strategy import Strategy


class ModelTester:
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


# estimator, useen feature data, unseen target data, scoring function.
