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
                    "max_depth": {"start": 1, "stop": 5, "step": 1},
                    "min_samples_split": {"start": 2, "stop": 10, "step": 1},
                    "min_samples_leaf": {"start": 1, "stop": 5, "step": 1},
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
                        "step": 1,
                    },  # type: int | None
                    "min_impurity_decrease": {
                        "start": 0.0,
                        "stop": 0.1,
                        "step": 0.01,
                    },
                    "ccp_alpha": {"start": 0.0, "stop": 0.5, "step": 0.01},
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
                        "step": 10,
                    },  # type: int | float | None
                },
                "ParamGridGaussianNaiveBayes": {},  # NOTE - There are only two hyperparameters that we cannot change! - This is left empty
                "ParamGridNeuralNetwork": {
                    "hidden_layer_sizes": {
                        "layers": {"start": 2, "stop": 10, "step": 1},
                        "layer_size": {"start": 2, "stop": 25, "step": 10},
                        # "input_layer": DEFINE AT RUNTIME
                        "output_layer": {
                            "start": 2,
                            "stop": 2,
                            "step": 1,
                        },  # 2 for binary classification
                    },
                    "activation": [
                        "logistic",
                        "relu",
                        "tanh",
                    ],  # type: Literal["identity", "logistic", "tanh", "relu"]
                    "solver": [
                        "sgd",
                        "lbfgs",
                        "adam",
                    ],  # type: Literal["lbfgs", "sgd", "adam"]
                    "learning_rate": [
                        "constant"
                    ],  # type: Literal["constant", "invscaling", "adaptive"]
                    "learning_rate_init": [0.001],
                    "alpha": {"start": 0.0001, "stop": 0.001, "step": 0.0001},
                    "max_iter": {"start": 1000, "stop": 10000, "step": 1000},
                    "tol": {"start": 0.0001, "stop": 0.001, "step": 0.0001},
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
                        "dist_params": {"low": 1, "high": 25, "size": 100},
                    },
                    "min_samples_split": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 2, "high": 10, "size": 100},
                    },
                    "min_samples_leaf": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 1, "high": 10, "size": 100},
                    },
                    "min_weight_fraction_leaf": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 0, "high": 1, "size": 10},
                    },
                    "max_features": [
                        "sqrt",
                        "log2",
                    ],  # type: Litteral["sqrt", "log2"] | int | float | None
                    "max_leaf_nodes": {
                        "dist": VariableDistribution.RANDINT.name,
                        "dist_params": {"low": 2, "high": 10, "size": 10},
                    },  # type: int | None
                    "min_impurity_decrease": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0.0, "high": 0.1, "size": 100},
                    },
                    "ccp_alpha": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0.0, "high": 0.5, "size": 10},
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
                        "dist_params": {"low": 10, "high": 500, "size": 10},
                    },
                },
                "RandomParamGridGaussianNaiveBayes": {},  # NOTE - There are only two hyperparameters that we cannot change! - This is left empty
                "RandomParamGridNeuralNetwork": {
                    "hidden_layer_sizes": {
                        "layers": {"start": 1, "stop": 10, "step": 1},
                        "layer_size": {"low": 2, "high": 10, "size": 2},
                        # "input_layer": DEFINE AT RUNTIME
                        "output_layer": {
                            "start": 2,
                            "stop": 2,
                            "step": 1,
                        },  # 2 for binary classification
                    },
                    "activation": [
                        "logistic",
                        "relu",
                        "tanh",
                    ],  # type: Literal["identity", "logistic", "tanh", "relu"]
                    "solver": [
                        "sgd",
                        "lbfgs",
                        "adam",
                    ],  # type: Literal["lbfgs", "sgd", "adam"]
                    "learning_rate": [
                        "constant",
                        "adaptive",
                    ],  # type: Literal["constant", "invscaling", "adaptive"]
                    "learning_rate_init": [0.001],
                    "alpha": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0.0001, "high": 0.001, "size": 100},
                    },
                    "max_iter": [1000],
                    "tol": {
                        "dist": VariableDistribution.RANDFLOAT.name,
                        "dist_params": {"low": 0.0001, "high": 0.001, "size": 100},
                    },
                },
            },
        }
