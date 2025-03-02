from typing import Self


from modules.config.utils.config_enums import (
    VariableDistribution,
)


class GridTemplate(object):
    """
    Singleton that defines the structure/layout of our configuration file.
    This layout is called low "template".
    """

    _instance = None

    # This is low singleton class since we only want 1 instance of low GridTemplate at all times
    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # We assign the template as an instance variable, however, seeing as the class is low singleton, it could've been low class variable as well.
            # Assigning it as an instance variable makes VSCode not recognise it as defined, however it runs just fine.
            cls._instance._template = cls._createTemplate()
        return cls._instance

    def getTemplate(self) -> dict:
        return self._template

    @classmethod
    def _createTemplate(self) -> dict:
        return {
            "ParamGrid": {
                "ParamGridDecisionTree": {
                    "criterion": [
                        "gini",
                        "log_loss",
                    ],  # type: Literal["gini", "entropy", "log_loss"]
                    "max_depth": {"start": 1, "stop": 25, "step": 5},
                    "min_samples_split": {"start": 2, "stop": 10, "step": 2},
                    "min_samples_leaf": {"start": 1, "stop": 10, "step": 2},
                    "min_weight_fraction_leaf": {
                        "start": 0.0,
                        "stop": 0.5,
                        "step": 0.1,
                    },
                    "max_features": [
                        "sqrt",
                        "log2",
                    ],  # type: Litteral["sqrt", "log2"] | int | float | None
                    "max_leaf_nodes": {
                        "start": 2,
                        "stop": 10,
                        "step": 2,
                    },  # type: int | None
                    "min_impurity_decrease": {
                        "start": 0.0,
                        "stop": 0.1,
                        "step": 0.02,
                    },
                    "ccp_alpha": {"start": 0.0, "stop": 0.5, "step": 0.05},
                },
                "ParamGridRandomForest": {
                    "n_estimators": {"start": 100, "stop": 1000, "step": 100},
                    "bootstrap": [True],
                    "oob_score": [
                        False
                    ],  # type: bool | Callable # TODO: Add score function
                    "max_samples": {
                        "start": 10,
                        "stop": 500,
                        "step": 50,
                    },  # type: int | float | None
                },
                "ParamGridCategoricalNaiveBayes": {"min_categories": [101]},
                "ParamGridNeuralNetwork": {
                    "hidden_layer_sizes": {
                        "layers": {"start": 2, "stop": 8, "step": 1},
                        "layer_size": [10, 15, 20, 25],
                    },
                    "activation": [
                        "logistic",
                        "relu",
                        "tanh",
                    ],  # type: Literal["logistic", "tanh", "relu"]
                    "solver": [
                        "sgd",
                        "adam",
                    ],  # type: Literal["sgd", "adam"]
                    "learning_rate": [
                        "adaptive"  # type: Literal["constant"]
                    ],
                    "learning_rate_init": [0.1],
                    "batch_size": [2000],
                    "alpha": [0.0001, 0.001, 0.01, 0.1],
                    "max_iter": [10000],
                    "tol": [0.0001],
                    "n_iter_no_change": [20],
                },
            },
            "RandomParamGrid": {
                "RandomParamGridDecisionTree": {
                    "criterion": [
                        "gini",
                        "log_loss",
                    ],  # type: Literal["gini", "entropy", "log_loss"]
                    "max_depth": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 3, "high": 25, "size": 100},
                    },
                    "min_samples_split": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 2, "high": 20, "size": 100},
                    },
                    "min_samples_leaf": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 1, "high": 10, "size": 100},
                    },
                    "min_weight_fraction_leaf": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0, "high": 0.5, "size": 100},
                    },
                    "max_features": [
                        "sqrt",
                        "log2",
                    ],  # type: Litteral["sqrt", "log2"] | int | float | None
                    "max_leaf_nodes": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 2, "high": 10, "size": 100},
                    },  # type: int | None
                    "min_impurity_decrease": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0.0, "high": 0.1, "size": 100},
                    },
                    "ccp_alpha": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0.0, "high": 0.5, "size": 50},
                    },
                },
                "RandomParamGridRandomForest": {
                    "n_estimators": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 100, "high": 1000, "size": 100},
                    },
                    "bootstrap": [True],
                    "oob_score": [
                        False
                    ],  # type: bool | Callable # TODO: Add score function
                    "max_samples": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 10, "high": 500, "size": 50},
                    },
                },
                "RandomParamGridCategoricalNaiveBayes": {"min_categories": [101]},
                "RandomParamGridNeuralNetwork": {
                    "hidden_layer_sizes": {
                        "layers": {"start": 1, "stop": 10, "step": 1},
                        "layer_size": [10, 15, 20, 25],
                    },
                    "activation": [
                        "logistic",
                        "relu",
                        "tanh",
                    ],  # type: Literal["logistic", "tanh", "relu"]
                    "solver": [
                        "sgd",
                        "adam",
                    ],  # type: Literal["sgd", "adam"]
                    "learning_rate": [
                        "adaptive",  # type: Literal["constant", "adaptive"]
                    ],
                    "learning_rate_init": [0.1],
                    "batch_size": [2000],
                    "alpha": [0.0001, 0.001, 0.01, 0.1],
                    "max_iter": [10000],
                    "tol": [0.0001],
                    "n_iter_no_change": [20],
                },
            },
        }
