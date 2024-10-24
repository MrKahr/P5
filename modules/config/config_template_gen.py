from typing import Self

from modules.config.config_enums import (
    CrossValidator,
    FeatureScoreFunc,
    FeatureSelectionCriterion,
    Model,
    LogLevel,
    ImputationMethod,
    NormalisationMethod,
    OutlierRemovalMethod,
    ModelScoreFunc,
    TrainingMethod,
)


class ConfigTemplate(object):
    """Singleton that defines  configuration template for the project.
    Note: We added additional params/longer attribute accesses for clarity."""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._template = (
                cls._createTemplate()
            )  # Added field access for clariy - VS-code cares, Python doesn't
        return cls._instance

    def getTemplate(self) -> dict:
        return self._template

    @classmethod
    def _createTemplate(self) -> dict:
        return {
            "General": {
                "loglevel": LogLevel.DEBUG.name,
                "n_jobs": -1,  # type: int | None  # NOTE: -1 means use all cores and None means 1 unless in joblib context
                "UseCleaner": True,
                "UseFeatureSelector": True,
                "UseTransformer": True,
                "UseOutlierRemoval": True,
            },
            "DataPreprocessing": {
                "Cleaning": {
                    "DeleteNanColumns": True,
                    "DeleteNonfeatures": False,
                    "DeleteMissingValues": False,
                    "DeleteUndeterminedValue": False,
                    "RemoveFeaturelessRows": True,
                    "RemoveFeaturelessRowsArgs": 3,
                    "FillNan": True,
                    "ShowNan": True,
                },
                "OutlierAnalysis": {
                    "OutlierRemovalMethod": OutlierRemovalMethod.ODIN.name,
                    "odinParams": {
                        "k": 30,
                        "T": 0,
                    },  # {number of neighbors, indegree threshold}
                    "avfParams": {"k": 10},  # {number of outliers to detect}
                },
                "Transformer": {
                    "OneHotEncodeLabels": [],  # type: list[str]
                    "ImputationMethod": ImputationMethod.KNN.name,
                    "NearestNeighbors": 5,
                    "NormalisationMethod": NormalisationMethod.MIN_MAX.name,
                    "NormaliseFeatures": [],  # type: list[str]
                },
                "FeatureSelection": {
                    "score_functions": [FeatureScoreFunc.CHI2.name],
                    "MutualInfoClassifArgs": {
                        "discrete_features": True,
                        "n_neighbors": 3,
                        "random_state": 12,
                    },
                    "GenericUnivariateSelect": True,
                    "GenericUnivariateSelectArgs": {
                        "mode": FeatureSelectionCriterion.PERCENTILE.name,
                        "param": 5,  # type: int | float | str  # The parameter for the mode
                    },
                    "ComputeFeatureCorrelation": True,
                    "VarianceThreshold": True,
                    "checkOverfitting": True,
                },
            },
            "ModelSelection": {
                "model": Model.DECISION_TREE.name,
                "DecisionTree": {
                    "criterion": "entropy",  # type: Literal["gini", "entropy", "log_loss"]
                    "max_depth": None,  # type: int | None
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "min_weight_fraction_leaf": 0,  # type: int | float
                    "max_features": None,  # type: int | None
                    "random_state": 42,  # type: int | None
                    "max_leaf_nodes": None,  # type: int | None
                    "min_impurity_decrease": 0.0,
                    "ccp_alpha": 0.0,
                },
                "RandomForest": {  # NOTE: DecisionTree arguments are also used for RandomForest
                    "n_estimators": 100,
                    "bootstrap": True,
                    "oob_score": False,  # type: bool | Callable # TODO: Add score function
                    "random_state": 53,  # type: int | None
                    "max_samples": None,  # type: int | float | None
                },
                "GaussianNaiveBayes": {},  # TODO: Maybe use CategoricalNaiveBayes instead
            },
            "CrossValidationSelection": {
                "cross_validator": CrossValidator.STRATIFIED_KFOLD.name,  # type: CrossValidator | None
                "StratifiedKFold": {
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 177,  # type: int | None  # NOTE: If shuffle is false, random_state must be None
                },
                "TimeSeriesSplit": {
                    "n_splits": 5,
                    "max_train_size": None,  # type: int | None
                    "test_size": None,  # type: int | None
                    "gap": 0,
                },
            },
            "ModelTraining": {
                "training_method": TrainingMethod.FIT.name,
                "PermutationFeatureImportance": {
                    "scoring": "",  # TODO: Add enum
                    "n_repeats": 10,
                    "random_state": 298,  # type: int | None
                },
                "RFE": {
                    "n_features_to_select": None,  # type: float | int | None  # NOTE: If None, half of the features are selected. If float between 0 and 1, it is the fraction of features to select.
                    "step": 1,  # type: float | int  # NOTE: If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
                },
                "RFECV": {
                    "min_features_to_select": 1,  # type: int
                    "step": 1,  # type: float | int
                },
                "RandomizedSearchCV": {
                    "n_iter": 10,  # NOTE: Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
                    "random_state": 378,
                },
                "GridSearchCV": {  # NOTE: GridSearch arguments are also used for RandomSearch
                    "refit": True,  # type: bool | str | Callable  # NOTE: For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.
                    "return_train_score": False,  # NOTE: Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
                },
            },
            "ModelScoreFunctions": {
                "score_functions": [ModelScoreFunc.THRESHOLD.name],
                "score_function_params": {"threshold": 20},
            },
            "ModelEvaluation": {"test5": ""},
        }
