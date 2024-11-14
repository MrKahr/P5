import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
)
from modules.config.config import Config
from modules.scoreFunctions.score_function_selector import ScoreFunctionSelector
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
        pipeline_report: dict,
    ):
        """
        Evaluates `estimator`'s performance on training and test datasets
        using various metrics and adds the results to the pipeline report.

        Parameters
        ----------
        estimator : FittedEstimator
            A fitted estimator which to evaluate.

        train_x : pd.DataFrame
            Training feature(s).

        train_true_y : pd.Series
            Target training feature, i.e., "Dag".

        test_x : pd.DataFrame
            Testing feature(s).

        test_true_y : pd.Series
            Target test feature, i.e., "Dag".

        pipeline_report : dict
            The pipeline report containing relevant results for the entire pipeline.
        """
        self._config = Config()
        self._estimator = estimator
        self._train_x = train_x
        self._train_true_y = train_true_y
        self._test_x = test_x
        self._test_true_y = test_true_y
        self._pipeline_report = pipeline_report

    def run(self) -> dict:
        """
        Evaluates `estimator`'s performance on training and test datasets
        using various metrics and adds the results to the pipeline report.

        Returns
        -------
        dict
            The pipeline report with evaluation results added.
        """
        logger.info(f"Testing {type(self._estimator).__name__} model")
        _avg = "weighted"

        # Compute train stats
        train_pred_y = self._estimator.predict(self._train_x)
        # FIXME: Make correct confusion matrix
        confusion_matrix_ = confusion_matrix(self._train_true_y, train_pred_y)

        # Get metrics to determine model accuracy for train set
        (
            train_true_negative,
            train_false_positive,
            train_false_negative,
            train_true_positive,
        ) = (
            confusion_matrix_[0][0],
            confusion_matrix_[0][1],
            confusion_matrix_[1][0],
            confusion_matrix_[1][1],
        )

        # Compute model evaluation metrics for train set
        train_precision = precision_score(
            self._train_true_y, train_pred_y, average=_avg, zero_division=np.nan
        )
        train_recall = recall_score(
            self._train_true_y, train_pred_y, average=_avg, zero_division=np.nan
        )
        train_tn_fn = train_true_negative + train_false_negative
        train_specificity = (
            train_true_negative / train_tn_fn if train_tn_fn > 0 else np.nan
        )

        # Compute testing stats
        test_pred_y = self._estimator.predict(self._test_x)
        # FIXME: Make correct confusion matrix
        confusion_matrix_ = confusion_matrix(self._test_true_y, test_pred_y)
        # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_)
        # disp.plot()
        # plt.show()

        # Get metrics to determine model accuracy for test set
        (
            test_true_negative,
            test_false_positive,
            test_false_negative,
            test_true_positive,
        ) = (
            confusion_matrix_[0][0],
            confusion_matrix_[0][1],
            confusion_matrix_[1][0],
            confusion_matrix_[1][1],
        )

        # Compute model evaluation metrics for test set
        test_precision = precision_score(
            self._test_true_y, test_pred_y, average=_avg, zero_division=np.nan
        )

        test_recall = recall_score(
            self._test_true_y, test_pred_y, average=_avg, zero_division=np.nan
        )
        test_tn_fn = test_true_negative + test_false_negative
        test_specificity = test_true_negative / test_tn_fn if test_tn_fn > 0 else np.nan

        # Compute model accuracies on train and test using all selected scoring functions
        train_accuracies = {}
        test_accuracies = {}
        # FIXME: Not ideal as predictions are computed multiple times.
        for func_name, func in ScoreFunctionSelector.getScoreFuncsModel().items():
            logger.info(f"Computing model accuracy using '{func_name}'")
            train_accuracies |= {
                func_name: func(self._estimator, self._train_x, self._train_true_y)
            }

            test_accuracies |= {
                func_name: func(self._estimator, self._test_x, self._test_true_y)
            }

        self._pipeline_report |= {
            "train_true_positive": train_true_positive,  # type: float
            "train_true_negative": train_true_negative,  # type: float
            "train_false_positive": train_false_positive,  # type: float
            "train_false_negative": train_false_negative,  # type: float
            "train_accuracies": train_accuracies,  # type: list[dict[str, float]]
            "train_precision": train_precision,  # type: float
            "train_recall": train_recall,  # type: float
            "train_specificity": train_specificity,  # type: float
            "test_true_positive": test_true_positive,  # type: float
            "test_true_negative": test_true_negative,  # type: float
            "test_false_positive": test_false_positive,  # type: float
            "test_false_negative": test_false_negative,  # type: float
            "test_accuracies": test_accuracies,  # type: list[dict[str, float]]
            "test_precision": test_precision,  # type: float
            "test_recall": test_recall,  # type: float
            "test_specificity": test_specificity,  # type: float
            "train_pred_y": train_pred_y,  # type: ndarray
            "train_x": self._train_x,  # type: pd.DataFrame
            "train_true_y": self._train_true_y,  # type: pd.Series
            "test_pred_y": test_pred_y,  # type: ndarray
            "test_x": self._test_x,  # type: pd.DataFrame
            "test_true_y": self._test_true_y,  # type: pd.Series
            "confusion_matrix": confusion_matrix_,  # type: ndarray
            "estimator": self._estimator,  # type: FittedEstimator
        }

        return self._pipeline_report
