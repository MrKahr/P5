from typing import Self

from modules.config.config_enums import CrossValidator, Model


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
                "RandomForest": {
                    "n_estimators": 100,
                    "bootstrap": True,
                    "oob_score": False,  # type: bool | Callable # TODO: Add score function
                    "n_jobs": -1,
                    "random_state": 53,  # type: int | None
                    "max_samples": None,  # type: int | float | None
                },
            },
            "CrossValidationSelection": {
                "cross_validator": CrossValidator.STRATIFIED_KFOLD.name,  # type: CrossValidator | None
                "StratifiedKFold": {
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 177,  # type: int | None # NOTE: If shuffle is false, random_state must be None
                },
                "TimeSeriesSplit": {
                    "n_splits": 5,
                    "max_train_size": None,  # type: int | None
                    "test_size": None,  # type: int | None
                    "gap": 0,
                },
            },
            "ModelTraining": {"test3": ""},
            "ModelTesting": {"test4": ""},
            "ModelEvaluation": {"test5": ""},
        }
