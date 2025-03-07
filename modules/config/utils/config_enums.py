from enum import Enum


# SECTION General
class LogLevel(Enum):
    INFO = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


# SECTION Data Preprocessing
class OutlierRemovalMethod(Enum):
    ODIN = 0
    AVF = 1


class DiscretizeMethod(Enum):
    NONE = 0
    CHIMERGE = 1
    NAIVE = 2


class ImputationMethod(Enum):
    NONE = 0
    MODE = 1
    KNN = 2


class DistanceMetric(Enum):
    ZERO_ONE = 0
    MATRIX = 1


class NormalisationMethod(Enum):
    NONE = 0
    MIN_MAX = 1


# SECTION Feature selection
class FeatureSelectionCriterion(Enum):
    PERCENTILE = 1
    K_BEST = 2
    FPR = 3
    FDR = 4
    FWE = 5


class FeatureScoreFunc(Enum):
    CHI2 = 0
    ANOVA_F = 1
    MUTUAL_INFO_CLASSIFIER = 2


# SECTION Model Selection
class Model(Enum):
    DECISION_TREE = 0
    RANDOM_FOREST = 1
    NEURAL_NETWORK = 2
    NAIVE_BAYES = 3


# SECTION Cross-Validation Selection
class CrossValidator(Enum):
    STRATIFIED_KFOLD = 0


# SECTION Model Training
class TrainingMethod(Enum):
    FIT = 0
    CROSS_VALIDATION = 1
    RFECV = 3
    RANDOM_SEARCH = 4
    GRID_SEARCH = 5


class VariableDistribution(Enum):
    # TODO: Consider implementing support for multivariate distributions

    # Continous distributions
    RANDFLOAT = 1  # Uniform distribution

    # Discrete distributions
    RANDINT = 106


# SECTION Model Testing
class ModelScoreFunc(Enum):
    # Use all score functions available
    ALL = -1

    # Our own custom scoring functions
    THRESHOLD = 0
    DISTANCE = 1

    # Scoring functions for classifications
    EXACT_ACCURACY = 100
    BALANCED_ACCURACY = 101
