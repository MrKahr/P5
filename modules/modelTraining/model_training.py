# model training
from modelSelection import model_name_enum


def model_training(
    mname: model_name_enum.ModelName,
    estimator,
):
    match mname:
        case model_name_enum.ModelName.DECISIONTREE:
            return estimator.fit()
        case model_name_enum.ModelName.RANDOMFOREST:
            return estimator.fit()
        case model_name_enum.ModelName.SUPPORTVECTOR:
            return estimator.fit()
        case model_name_enum.ModelName.NAIVEBAYES:
            return estimator.fit()
