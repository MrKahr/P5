import os
from typing import Callable, TypeAlias, Union
from numpy.typing import ArrayLike, NDArray

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


StrPath: TypeAlias = str | os.PathLike[str]


UnfittedEstimator: TypeAlias = Union[
    DecisionTreeClassifier, RandomForestClassifier, GaussianNB
]
FittedEstimator: TypeAlias = Union[
    DecisionTreeClassifier, RandomForestClassifier, GaussianNB
]

CrossValidator: TypeAlias = Union[StratifiedKFold, TimeSeriesSplit]

ModelScoreCallable: TypeAlias = Callable[[FittedEstimator, ArrayLike, ArrayLike], float]

FeatureSelectScoreCallable: TypeAlias = Callable[
    [NDArray, NDArray], tuple[NDArray, NDArray] | NDArray
]
