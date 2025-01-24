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
                "UseStatisticalFeatureSelector": False,
                "UseTransformer": False,
                "UseOutlierRemoval": False,
                "UseContinuousFeatures": False,
            },
            "DataPreprocessing": {
                "Cleaning": {
                    "DeleteNanColumns": True,
                    "DeleteNonfeatures": True,
                    "DeleteMissingValues": False,  # Missing value = 2
                    "DeleteUndeterminedValue": False,  # Undetermined = 100
                    "RemoveNaNAmount": True,
                    "RemoveNaNAmountArgs": 3,
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
                    "Discretization": {
                        "DiscretizeColumns": [
                            "Sårrand (cm)",
                            "Midte (cm)",
                        ],  # type: list[str]
                        "DiscretizeMethod": DiscretizeMethod.NONE.name,
                        "ChiMergeMaximumMergeThreshold": {
                            "Sårrand (cm)": np.inf.hex(),  # Convert to string as not all JSON parsers support np.inf
                            "Midte (cm)": np.inf.hex(),
                        },
                        "DiscretizeDesiredIntervals": {
                            "Sårrand (cm)": 5,
                            "Midte (cm)": 5,
                        },
                    },
                    "OneHotEncoding": {
                        "UseOneHotEncoding": False,
                        "OneHotEncodeLabels": [
                            "Eksudattype",
                        ],  # type: list[str]
                    },
                    "Imputation": {
                        "ImputationMethod": ImputationMethod.NONE.name,
                        "KNN_NearestNeighbors": 5,
                        "KNN_DistanceMetric": DistanceMetric.MATRIX.name,
                    },
                    "Normalisation": {
                        "NormalisationMethod": NormalisationMethod.NONE.name,
                        "NormaliseFeatures": [],  # type: list[str]
                    },
                },
                "StatisticalFeatureSelection": {
                    "score_function": FeatureScoreFunc.MUTUAL_INFO_CLASSIFIER.name,
                    "MutualInfoClassifArgs": {
                        "discrete_features": False,  # False if dataset contains floats (i.e. if using MÅL)
                        "n_neighbors": 3,
                        "random_state": 12,
                    },
                    "GenericUnivariateSelect": True,
                    "GenericUnivariateSelectArgs": {
                        "mode": FeatureSelectionCriterion.PERCENTILE.name,
                        "param": 50,  # type: int | float | str  # The parameter for the mode
                    },
                },
            },
            "ModelSelection": {
                "model": Model.DECISION_TREE.name,
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
                "CategoricalNaiveBayes": {
                    "min_categories": 101  # NOTE: Should be largest value + 1 in dataset to prevent index out of bounds
                },
                "NeuralNetwork": {
                    "hidden_layer_sizes": (10, 10),
                    "activation": "relu",  # type: Literal["logistic", "tanh", "relu"]
                    "solver": "adam",  # type: Literal["sgd", "adam"]
                    "learning_rate": "adaptive",  # type: Literal["constant", "adaptive"]
                    "learning_rate_init": 0.1,
                    "batch_size": 2000,
                    "alpha": 0.001,
                    "max_iter": 10000,
                    "tol": 0.0001,
                    "n_iter_no_change": 20,
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
            },
            "ModelTraining": {
                "training_method": TrainingMethod.FIT.name,  # NOTE: ensure param set to FIT when first generating config.
                "score_functions": [ModelScoreFunc.ALL.name],
                "score_function_params": {
                    "threshold": 20,
                },
                "score_function_weights": {
                    ModelScoreFunc.THRESHOLD.name.lower(): 0.8,
                    ModelScoreFunc.DISTANCE.name.lower(): 0.9,
                    ModelScoreFunc.EXACT_ACCURACY.name.lower(): 1,
                    ModelScoreFunc.BALANCED_ACCURACY.name.lower(): 1.1,
                },
                "train_test_split": {
                    "random_state": 111,
                    "train_size": 0.80,  # NOTE: Percentage of dataset used for training
                },
                "PermutationFeatureImportance": {
                    "n_repeats": 10,  # NOTE: Use 500 for model evaluation
                    "random_state": 298,  # type: int | None
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
                    "return_train_score": False,  # NOTE: Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
                    "refit": ModelScoreFunc.EXACT_ACCURACY.name.lower(),  # NOTE: Its semantics is different from the score function as it is specific for param Search results
                    "verbose": 1,  # type: Literal[0, 1, 2, 3]  # NOTE: 0 = silent, 1 = the computation time for each fold and parameter candidate is displayed, 2 = the score is also displayed, 3 = the fold and candidate parameter indexes are also displayed.
                },
            },
            "ModelEvaluation": {
                "print_model_report": True,
                "plot_confusion_matrix": True,
                "plot_roc_curves": True,
                "plot_feature_importance": True,
                "plot_tree": True,
                "plot_decision_boundary": False,  # NOTE: Half-baked implementation
            },
        }
