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
    CHIMERGE = 0
    NAIVE = 1


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
    MUTUAL_INFO_CLASSIFER = 2


# SECTION Model Selection
class Model(Enum):
    DECISION_TREE = 0
    RANDOM_FOREST = 1
    NEURAL_NETWORK = 2
    NAIVE_BAYES = 3


# SECTION Cross-Validation Selection
class CrossValidator(Enum):
    STRATIFIED_KFOLD = 0
    TIMESERIES_SPLIT = 1


# SECTION Model Training
class TrainingMethod(Enum):
    FIT = 0
    CROSS_VALIDATION = 1
    RFE = 2
    RFECV = 3
    RANDOM_SEARCH_CV = 4
    GRID_SEARCH_CV = 5


class VariableDistribution(Enum):
    # TODO: Consider implementing support for multivariate distributions

    # Continous distributions
    UNIFORM = 1

    # Discrete distributions
    BERNOULLI = 100
    BINOM = 101
    GEOM = 103
    HYPERGEOM = 104
    POISSON = 105
    RANDINT = 106  # Uniform distribution
    RANDFLOAT = 107
    ZIPF = 108


# SECTION Model Testing
class ModelScoreFunc(Enum):
    # Use all score functions available
    ALL = -1

    # Our own custom scoring functions
    THRESHOLD = 0
    DISTANCE = 1

    # Scoring functions for classifications
    ACCURACY = 100
    BALANCED_ACCURACY = 101

    # Scoring functions for clustering
    ADJUSTED_MUTUAL_INFO_SCORE = 200

    # Scoring functions for regression
    EXPLAINED_VARIANCE = 300
