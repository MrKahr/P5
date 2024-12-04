from typing import Self

import numpy as np


from modules.config.utils.config_enums import (
    CrossValidator,
    DiscretizeMethod,
    DistanceMetric,
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
    """
    Singleton that defines the structure/layout of our configuration file.
    This layout is called a "template".
    """

    _instance = None

    # This is a singleton class since we only want 1 instance of a ConfigTemplate at all times
    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # We assign the template as an instance variable, however, seeing as the class is a singleton, it could've been a class variable as well.
            # Assigning it as an instance variable makes VSCode not recognise it as defined, however it runs just fine.
            cls._instance._template = cls._createTemplate()
        return cls._instance

    def getTemplate(self) -> dict:
        return self._template

    @classmethod
    def _createTemplate(self) -> dict:
        return {
            "General": {
                "loglevel": LogLevel.DEBUG.name,
                "n_jobs": -1,  # type: int | None  # NOTE: -1 means use all cores and None means 1 unless in joblib context
                "write_figure_to_disk": True,
                "UseCleaner": True,
                "UseFeatureSelector": False,
                "UseTransformer": False,
                "UseOutlierRemoval": False,
                "UseContinuousFeatures": False,
            },
            "DataPreprocessing": {
                "Cleaning": {
                    "DeleteNanColumns": True,
                    "DeleteNonfeatures": True,  # TODO: Remove from config and hardcode True in cleaner
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
                    "DiscretizeColumns": [
                        "Sårrand (cm)",
                        "Midte (cm)",
                    ],  # type: list[str]
                    "DiscretizeMethod": DiscretizeMethod.NAIVE.name,
                    "ChiMergeMaximumMergeThreshold": {
                        "Sårrand (cm)": np.inf,
                        "Midte (cm)": np.inf,
                    },
                    "DiscretizeDesiredIntervals": {
                        "Sårrand (cm)": -1,
                        "Midte (cm)": -1,
                    },
                    "UseOneHotEncoding": False,
                    "OneHotEncodeLabels": [],  # type: list[str]
                    "ImputationMethod": ImputationMethod.NONE.name,
                    "KNN_NearestNeighbors": 5,
                    "KNN_DistanceMetric": DistanceMetric.MATRIX.name,
                    "NormalisationMethod": NormalisationMethod.NONE.name,
                    "NormaliseFeatures": [],  # type: list[str]
                },
                "FeatureSelection": {
                    "score_function": FeatureScoreFunc.CHI2.name,
                    "MutualInfoClassifArgs": {
                        "discrete_features": True,
                        "n_neighbors": 3,
                        "random_state": 12,
                    },
                    "GenericUnivariateSelect": True,
                    "GenericUnivariateSelectArgs": {
                        "mode": FeatureSelectionCriterion.PERCENTILE.name,
                        "param": 25,  # type: int | float | str  # The parameter for the mode
                    },
                    "ComputeFeatureCorrelation": True,
                    "VarianceThreshold": True,
                    "checkOverfitting": True,
                },
            },
            "ModelSelection": {
                "model": Model.RANDOM_FOREST.name,
                "DecisionTree": {
                    "criterion": "gini",  # type: Literal["gini", "entropy", "log_loss"]
                    "max_depth": None,  # type: int | None
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "min_weight_fraction_leaf": 0,  # type: int | float
                    "max_features": "sqrt",  # type: Litteral["sqrt", "log2"] | int | float | None
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
                "NeuralNetwork": {
                    "hidden_layer_sizes": (20, 2),
                    "activation": "logistic",  # type: Literal["identity", "logistic", "tanh", "relu"]
                    "solver": "sgd",  # type: Literal["lbfgs", "sgd", "adam"]
                    "learning_rate": "constant",  # type: Literal["constant", "invscaling", "adaptive"]
                    "learning_rate_init": 0.001,
                    "alpha": 0.0001,
                    "max_iter": 1000,
                    "tol": 0.0001,
                    "random_state": 678,
                },
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
                "training_method": TrainingMethod.GRID_SEARCH_CV.name,
                "score_functions": [ModelScoreFunc.ALL.name],
                "score_function_params": {
                    "threshold": 20,
                },
                "score_function_weights": {
                    "threshold": 0.8,
                    "distance": 0.9,
                    "accuracy": 1,
                    "balanced_accuracy": 1.1,
                },
                "PermutationFeatureImportance": {
                    "n_repeats": 10,
                    "random_state": 298,  # type: int | None
                },
                "RFE": {
                    "n_features_to_select": 0.50,  # type: float | int | None  # NOTE: If None, half of the features are selected. If float between 0 and 1, it is the fraction of features to select.
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
                "GridSearchCV": {
                    "refit": "accuracy",  # type: bool | str | Callable  # NOTE: For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.
                    "return_train_score": False,  # NOTE: Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
                    "verbose": 1,  # type: Literal[0, 1, 2, 3]  # NOTE: 0 = silent, 1 = the computation time for each fold and parameter candidate is displayed, 2 = the score is also displayed, 3 = the fold and candidate parameter indexes are also displayed.
                },
            },
            "ModelEvaluation": {
                "print_model_report": True,
                "plot_confusion_matrix": True,
                "plot_roc_curves": True,
                "plot_tree": True,
            },
        }