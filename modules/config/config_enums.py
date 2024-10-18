from enum import Enum


class Model(Enum):
    DECISION_TREE = 0
    RANDOM_FOREST = 1
    NEURAL_NETWORK = 2
    NAIVE_BAYES = 3
    SUPPORT_VECTOR = 4


class CrossValidator(Enum):
    STRATIFIED_KFOLD = 0
    TIMESERIES_SPLIT = 1
