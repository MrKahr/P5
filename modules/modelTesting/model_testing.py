import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)
from modules.config.config import Config
from modules.modelTesting.score_functions import ScoreFunctions
from modules.types import FittedEstimator
from modules.logging import logger


class ModelTester:
    def __init__(
        self,
        estimator: FittedEstimator,
        train_x: pd.DataFrame,
        train_true_y: pd.Series,
        test_x: pd.DataFrame,
        test_true_y: pd.Series,
        model_report: dict,
    ):
        self._config = Config()
        self._estimator = estimator
        self._train_x = train_x
        self._train_true_y = train_true_y
        self._test_x = test_x
        self._test_true_y = test_true_y
        self._model_report = model_report

    def run(self) -> dict:
        logger.info(f"Testing {type(self._estimator).__name__} model")
        _avg = "weighted"

        # Compute train stats
        train_pred_y = self._estimator.predict(self._train_x)
        confusion_matrix_ = confusion_matrix(self._train_true_y, train_pred_y)

        train_tn, train_fp, train_fn, train_tp = (
            confusion_matrix_[0][0],
            confusion_matrix_[0][1],
            confusion_matrix_[1][0],
            confusion_matrix_[1][1],
        )

        train_precision = precision_score(
            self._train_true_y, train_pred_y, average=_avg, zero_division=np.nan
        )

        train_recall = recall_score(
            self._train_true_y, train_pred_y, average=_avg, zero_division=np.nan
        )
        train_tn_fn = train_tn + train_fn
        train_specificity = train_tn / train_tn_fn if train_tn_fn > 0 else np.nan

        # Compute testing stats
        test_pred_y = self._estimator.predict(self._test_x)
        confusion_matrix_ = confusion_matrix(self._test_true_y, test_pred_y)

        test_tn, test_fp, test_fn, test_tp = (
            confusion_matrix_[0][0],
            confusion_matrix_[0][1],
            confusion_matrix_[1][0],
            confusion_matrix_[1][1],
        )

        test_precision = precision_score(
            self._test_true_y, test_pred_y, average=_avg, zero_division=np.nan
        )

        test_recall = recall_score(
            self._test_true_y, test_pred_y, average=_avg, zero_division=np.nan
        )
        test_tn_fn = test_tn + test_fn
        test_specificity = test_tn / test_tn_fn if test_tn_fn > 0 else np.nan

        # Compute model accuracies on train and test using all selected scoring functions
        train_accuracies = {}
        test_accuracies = {}
        # FIXME: Not ideal as predictions are computed multiple times.
        for func_name, func in ScoreFunctions.getScoreFuncsModel().items():
            logger.info(f"Computing model accuracy using '{func_name}'")
            train_accuracies |= {
                func_name: func(self._estimator, self._train_x, self._train_true_y)
            }

            test_accuracies |= {
                func_name: func(self._estimator, self._test_x, self._test_true_y)
            }

        self._model_report |= {
            "train_true_positive": train_tp,  # type: float
            "train_true_negative": train_tn,  # type: float
            "train_false_positive": train_fp,  # type: float
            "train_false_negative": train_fn,  # type: float
            "train_accuracies": train_accuracies,  # type: list[dict[str, float]]
            "train_precision": train_precision,  # type: float
            "train_recall": train_recall,  # type: float
            "train_specificity": train_specificity,  # type: float
            "test_true_positive": test_tp,  # type: float
            "test_true_negative": test_tn,  # type: float
            "test_false_positive": test_fp,  # type: float
            "test_false_negative": test_fn,  # type: float
            "test_accuracies": test_accuracies,  # type: list[dict[str, float]]
            "test_precision": test_precision,  # type: float
            "test_recall": test_recall,  # type: float
            "test_specificity": test_specificity,  # type: float
        }

        return self._model_report
