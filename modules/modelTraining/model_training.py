# model training
from modelSelection import model_name_enum
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier


def model_training(
    mname: model_name_enum.ModelName,
    estimator: DecisionTreeClassifier,
    trainingMethod: str,
    scoringFunction,
    Xtrain,
    Ytrain,
):
    match trainingMethod:
        case "GridSearchCV":
            return GridSearchCV(estimator, scoring=scoringFunction)
        case "RandomizedSearchCV":
            return RandomizedSearchCV(estimator, scoring=scoringFunction)
        case "RecursiveFeatureValidation":
            return RFE(estimator, scoring=scoringFunction)
        case "RecursiveFeatureValidationCV":
            return RFECV(estimator, scoring=scoringFunction)
        case "fit":
            return estimator.fit(X=Xtrain, Y=Ytrain, scoring=scoringFunction)
