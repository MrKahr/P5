from numpy.typing import NDArray
from sklearn.base import accuracy_score
from sklearn.metrics import (
    balanced_accuracy_score,
    explained_variance_score,
    make_scorer,
)

from modules.config.config import Config
from modules.config.config_enums import ModelScoreFunc
from modules.logging import logger
from modules.types import FittedEstimator, ModelScoreCallable


class ScoreFunctions:
    _config = Config()
    _cache = None  # Score function cache

    @classmethod
    def _threshold(
        cls,
        estimator: FittedEstimator,
        x_train: NDArray,
        y_target: NDArray,
        threshold: int,
    ) -> float:
        predictions = estimator.predict(x_train)

        score = 0
        for i, prediction in enumerate(predictions):
            if y_target[i] >= threshold:
                score += 1
            elif prediction < threshold:
                score += 1

        return float(score / len(predictions))

    @classmethod
    def _distance(
        cls, estimator: FittedEstimator, x_train: NDArray, y_target: NDArray
    ) -> float:
        predictions = estimator.predict(x_train)
        max_y = y_target[y_target.argmax()]

        score = 0
        for i, prediction in enumerate(predictions):
            score += 1 - abs(prediction - y_target[i]) / max_y

        return float(score / len(predictions))

    @classmethod
    def getModelScoreFuncs(cls) -> dict[str, ModelScoreCallable]:
        if cls._cache is None:
            parent_key = "ModelScoreFunctions"
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
                                threshold=cls._config.getValue("threshold", parent_key),
                            )
                        }
                    case ModelScoreFunc.DISTANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.DISTANCE.name.lower(): cls._distance
                        }
                    case ModelScoreFunc.ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.ACCURACY.name.lower(): accuracy_score
                        }
                    case ModelScoreFunc.BALANCED_ACCURACY.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.BALANCED_ACCURACY.name.lower(): balanced_accuracy_score
                        }
                    case ModelScoreFunc.EXPLAINED_VARIANCE.name:
                        selected_score_funcs |= {
                            ModelScoreFunc.EXPLAINED_VARIANCE.name.lower(): explained_variance_score
                        }
                    case _:
                        raise TypeError(
                            f"Invalid score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                        )

            logger.info(f"Using model score functions: '{selected_score_funcs.keys()}'")
            cls._cache = selected_score_funcs

        return next(
            iter(cls._cache.values())
        )  # FIXME: Temporary solution until multi-score is implemented
