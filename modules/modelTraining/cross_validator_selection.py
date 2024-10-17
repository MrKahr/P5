# cross_vaidator_selection method takes CrossValidatorName enum as option, returns CrossValidator of the chosen type.

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from modules.modelTraining.cross_validator_enum import CrossValidatorName


def cross_validator_selection(cvname: CrossValidatorName):
    match cvname:
        case CrossValidatorName.STRATIFIEDKFOLD:
            print("StratifiedKFold")
            return StratifiedKFold(n_splits=5)
        case CrossValidatorName.TIMESERIES:
            print("TimeseriesSplit")
            return TimeSeriesSplit(n_splits=5)
