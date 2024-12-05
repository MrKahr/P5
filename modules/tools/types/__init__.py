import os
from typing import Callable, TypeAlias, Union
from numpy.typing import ArrayLike, NDArray

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from modules.gpuBackend.models.mlp_gpu import MLPClassifierGPU


StrPath: TypeAlias = str | os.PathLike[str]


UnfittedEstimator: TypeAlias = Union[
    DecisionTreeClassifier,
    RandomForestClassifier,
    GaussianNB,
    MLPClassifier,
    MLPClassifierGPU,
]
FittedEstimator: TypeAlias = Union[
    DecisionTreeClassifier,
    RandomForestClassifier,
    GaussianNB,
    MLPClassifier,
    MLPClassifierGPU,
]

CrossValidator: TypeAlias = Union[StratifiedKFold]

ModelScoreCallable: TypeAlias = Callable[[FittedEstimator, ArrayLike, ArrayLike], float]

FeatureSelectScoreCallable: TypeAlias = Callable[
    [NDArray, NDArray], tuple[NDArray, NDArray] | NDArray
]
