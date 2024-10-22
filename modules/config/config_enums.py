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


class TrainingMethod(Enum):
    FIT = 0
    CROSS_VALIDATION = 1
    RFE = 2
    RFECV = 3
    RANDOM_SEARCH_CV = 4
    GRID_SEARCH_CV = 5
