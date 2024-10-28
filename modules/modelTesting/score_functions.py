from typing import Optional
from numpy.typing import NDArray
from numpy import ndarray
import pandas as pd
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


# FIXME: Optimize code structure of class
class ScoreFunctions:
    _config = Config()
    _cache = {
        "model": None,
        "no_model": None,
    }  # Score function cache # TODO: Explain why it's here

    @classmethod
    def _threshold(
        cls,
        true_y: pd.Series | NDArray,
        pred_y: NDArray,
        threshold: int,
    ) -> float:
        score = 0
        if not isinstance(true_y, ndarray):
            true_y = true_y.to_numpy(copy=True)
        for i, prediction in enumerate(pred_y):
            if true_y[i] >= threshold:
                score += 1
            elif prediction < threshold:
                score += 1
        return float(score / len(pred_y))

    @classmethod
    def _distance(cls, true_y: pd.Series | NDArray, pred_y: NDArray) -> float:
        score = 0
        if not isinstance(true_y, ndarray):
            true_y = true_y.to_numpy(copy=True)
        max_y = true_y[true_y.argmax()]
        for i, prediction in enumerate(pred_y):
            score += 1 - abs(prediction - true_y[i]) / max_y
        return float(score / len(pred_y))

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
                if score_func in [
                    ModelScoreFunc.THRESHOLD.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.THRESHOLD.name.lower(): make_scorer(
                            cls._threshold,
                            threshold=cls._config.getValue(
                                "threshold", parent_key="score_function_params"
                            ),
                        )
                    }
                if score_func in [
                    ModelScoreFunc.DISTANCE.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.DISTANCE.name.lower(): make_scorer(cls._distance)
                    }
                if score_func in [
                    ModelScoreFunc.ACCURACY.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.ACCURACY.name.lower(): make_scorer(
                            accuracy_score
                        )
                    }
                if score_func in [
                    ModelScoreFunc.BALANCED_ACCURACY.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.BALANCED_ACCURACY.name.lower(): make_scorer(
                            balanced_accuracy_score
                        )
                    }
                if score_func in [
                    ModelScoreFunc.EXPLAINED_VARIANCE.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.EXPLAINED_VARIANCE.name.lower(): make_scorer(
                            explained_variance_score
                        )
                    }

                if not selected_score_funcs:
                    raise TypeError(
                        f"Invalid score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                    )

            logger.info(
                f"Using model score functions: '{", ".join(selected_score_funcs.keys())}'"
            )
            cls._cache["model"] = selected_score_funcs

        return cls._cache["model"]

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
                                estimator=estimator,
                                train_x=train_x,
                                true_y=true_y,
                                threshold=threshold,
                            ),
                        }
                    case ModelScoreFunc.DISTANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.DISTANCE.name.lower(): lambda train_x, true_y, estimator=estimator: cls._distance(
                                estimator=estimator, train_x=train_x, true_y=true_y
                            )
                        }
                    case ModelScoreFunc.ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.ACCURACY.name.lower(): lambda train_x, true_y, estimator=estimator: cls._accuracy(
                                estimator=estimator, train_x=train_x, true_y=true_y
                            )
                        }
                    case ModelScoreFunc.BALANCED_ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.BALANCED_ACCURACY.name.lower(): lambda train_x, true_y, estimator=estimator: cls._balanced_accuracy(
                                estimator=estimator, train_x=train_x, true_y=true_y
                            )
                        }
                    case ModelScoreFunc.EXPLAINED_VARIANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.EXPLAINED_VARIANCE.name.lower(): lambda train_x, true_y, estimator=estimator: cls._explained_variance(
                                estimator=estimator, train_x=train_x, true_y=true_y
                            )
                        }
                    case _:
                        raise TypeError(
                            f"Invalid score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                        )
            cls._cache["no_model"] = selected_score_funcs

        return cls._cache["no_model"]
