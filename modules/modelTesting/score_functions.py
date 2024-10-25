from typing import Optional
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    make_scorer,
)

from modules.config.config import Config
from modules.config.config_enums import ModelScoreFunc
from modules.logging import logger
from modules.types import FittedEstimator, ModelScoreCallable, NoModelScoreCallable


class ScoreFunctions:
    _config = Config()
    _cache = {"model": None, "no_model": None}  # Score function cache

    @classmethod
    def _threshold(
        cls,
        estimator: FittedEstimator,
        train_x: NDArray,
        true_y: NDArray,
        threshold: int,
    ) -> float:
        pred_y = estimator.predict(train_x)

        score = 0
        for i, prediction in enumerate(pred_y):
            if true_y[i] >= threshold:
                score += 1
            elif prediction < threshold:
                score += 1

        return float(score / len(pred_y))

    @classmethod
    def _distance(
        cls, estimator: FittedEstimator, train_x: NDArray, true_y: NDArray
    ) -> float:
        pred_y = estimator.predict(train_x)
        max_y = true_y[true_y.argmax()]

        score = 0
        for i, prediction in enumerate(pred_y):
            score += 1 - abs(prediction - true_y[i]) / max_y

        return float(score / len(pred_y))

    @classmethod
    def _accuracy(
        cls, estimator: FittedEstimator, train_x: NDArray, true_y: NDArray
    ) -> float:
        pred_y = estimator.predict(train_x)
        return accuracy_score(true_y, pred_y)

    @classmethod
    def _balanced_accuracy(
        cls, estimator: FittedEstimator, train_x: NDArray, true_y: NDArray
    ) -> float:
        pred_y = estimator.predict(train_x)
        return balanced_accuracy_score(true_y, pred_y)

    @classmethod
    def _explained_variance(
        cls, estimator: FittedEstimator, train_x: NDArray, true_y: NDArray
    ) -> float:
        pred_y = estimator.predict(train_x)
        return explained_variance_score(true_y, pred_y)

    @classmethod
    def getScoreFuncsModel(
        cls,
    ) -> dict[str, ModelScoreCallable]:
        if cls._cache["model"] is None:
            parent_key = "ModelTraining"
            score_funcs = cls._config.getValue("score_functions", parent_key)
            if not isinstance(score_funcs, list):
                score_funcs = [score_funcs]

            selected_score_funcs = {}
            for score_func in score_funcs:
                match score_func:
                    case ModelScoreFunc.THRESHOLD.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.THRESHOLD.name.lower(): make_scorer(
                                cls._threshold,
                                threshold=cls._config.getValue(
                                    "threshold", parent_key="score_function_params"
                                ),
                            )
                        }
                    case ModelScoreFunc.DISTANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.DISTANCE.name.lower(): cls._distance
                        }
                    case ModelScoreFunc.ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.ACCURACY.name.lower(): cls._accuracy
                        }
                    case ModelScoreFunc.BALANCED_ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.BALANCED_ACCURACY.name.lower(): cls._balanced_accuracy
                        }
                    case ModelScoreFunc.EXPLAINED_VARIANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.EXPLAINED_VARIANCE.name.lower(): cls._explained_variance
                        }
                    case _:
                        raise TypeError(
                            f"Invalid score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                        )

            logger.info(
                f"Using model score functions: '{", ".join(selected_score_funcs.keys())}'"
            )
            cls._cache["model"] = selected_score_funcs

        return cls._cache

    @classmethod
    def getScoreFuncNoModel(
        cls, estimator: Optional[FittedEstimator] = None
    ) -> dict[str, NoModelScoreCallable]:
        if cls._cache is None:
            if estimator is None:
                raise ValueError(
                    f"A model is required when building the score cache (got '{estimator}')"
                )

            parent_key = "ModelTraining"
            score_funcs = cls._config.getValue("score_functions", parent_key)
            if not isinstance(score_funcs, list):
                score_funcs = [score_funcs]

            selected_score_funcs = {}
            for score_func in score_funcs:
                match score_func:
                    case ModelScoreFunc.THRESHOLD.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.THRESHOLD.name.lower(): lambda train_x, true_y, estimator=estimator, threshold=cls._config.getValue(
                                "threshold", parent_key="score_function_params"
                            ): cls._threshold(
                                estimator, train_x, true_y, threshold
                            ),
                        }
                    case ModelScoreFunc.DISTANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.DISTANCE.name.lower(): lambda train_x, true_y, estimator=estimator: cls._distance(
                                estimator, train_x, true_y
                            )
                        }
                    case ModelScoreFunc.ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.ACCURACY.name.lower(): lambda train_x, true_y, estimator=estimator: cls._accuracy(
                                estimator, train_x, true_y
                            )
                        }
                    case ModelScoreFunc.BALANCED_ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.BALANCED_ACCURACY.name.lower(): lambda train_x, true_y, estimator=estimator: cls._balanced_accuracy(
                                estimator, train_x, true_y
                            )
                        }
                    case ModelScoreFunc.EXPLAINED_VARIANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.EXPLAINED_VARIANCE.name.lower(): lambda train_x, true_y, estimator=estimator: cls._explained_variance(
                                estimator, train_x, true_y
                            )
                        }
                    case _:
                        raise TypeError(
                            f"Invalid score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                        )
            cls._cache["no_model"] = selected_score_funcs

        return cls._cache["no_model"]
