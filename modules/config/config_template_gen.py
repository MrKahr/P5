from typing import Self

from modules.config.config_enums import Model


class ConfigTemplate(object):
    """Singleton that defines as configuration template for the project.
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
                "loglevel": "DEBUG",
            },
            "DataPreprocessing": {"test1": ""},
            "ModelSelection": {
                "model": Model.DECISION_TREE.name,
                "DecisionTree": {
                    "criterion": "entropy",  # type: Literal["gini", "entropy", "log_loss"]
                    "max_depth": None,  # type: int
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "min_weight_fraction_leaf": 0,  # type: int | float
                    "max_features": None,  # type: int
                    "random_state": 42,
                    "max_leaf_nodes": None,  # type: int
                    "min_impurity_decrease": 0.0,
                    "ccp_alpha": 0.0,
                },
                "RandomForest": {
                    "n_estimators": 100,
                    "bootstrap": True,
                    "oob_score": False,  # type: bool | Callable # TODO: Add score function
                    "n_jobs": -1,
                    "random_state": 53,
                    "max_samples": None,  # type: int | float
                },
            },
            "ModelTraining": {"test3": ""},
            "ModelTesting": {"test4": ""},
            "ModelEvaluation": {"test5": ""},
        }
