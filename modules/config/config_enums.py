from enum import Enum


class Config:
    class General(Enum):
        SHOW_PLOT = 0

    class DataCleaning(Enum):
        DEL_MISSING_VALUE = 0

    class FeatureSelection(Enum):
        PERMUTATION = 0
