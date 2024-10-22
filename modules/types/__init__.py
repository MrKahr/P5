import os
from typing import TypeAlias, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


StrPath: TypeAlias = str | os.PathLike[str]


UnfittedEstimator: TypeAlias = Union[
    DecisionTreeClassifier, RandomForestClassifier, GaussianNB, SVC
]
FittedEstimator: TypeAlias = Union[
    DecisionTreeClassifier, RandomForestClassifier, GaussianNB, SVC
]

CrossValidator: TypeAlias = Union[StratifiedKFold, TimeSeriesSplit]
