from enum import Enum


class LogLevel(Enum):
    INFO = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class OutlierRemovalMethod(Enum):
    NONE = 0
    ODIN = 0
    AVF = 0


class ImputationMethod(Enum):
    NONE = 0
    MODE = 1
    KNN = 2


class NormalisationMethod(Enum):
    NONE = 0
    MIN_MAX = 1


class Model(Enum):
    DECISION_TREE = 0
    RANDOM_FOREST = 1
    NEURAL_NETWORK = 2
    NAIVE_BAYES = 3
    SUPPORT_VECTOR = 4


class CrossValidator(Enum):
    STRATIFIED_KFOLD = 0
    TIMESERIES_SPLIT = 1


class ScoreFunction(Enum):
    # Custom scoring functions
    CUSTOM_SCORE_FUNC = 0
    # Scoring functions for classifications
    ACCURACY = 100
    BALANCED_ACCURACY = 101
    # Scoring functions for clustering
    ADJUSTED_MUTUAL_INFO_SCORE = 200
    # Scoring functions for regression
    EXPLAINED_VARIANCE = 300


class Mode(Enum):
    PERCENTILE = 1
    K_BEST = 2
    FPR = 3
    FDR = 4
    FWE = 5
