import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
)
from modules.config.config import Config
from modules.scoreFunctions.score_function_selector import ScoreFunctionSelector
from modules.logging import logger


class ModelTester:
    def __init__(
        self,
        pipeline_report: dict,
    ):
        """
        Evaluates `estimator`'s performance on training and test datasets
        using various metrics and adds the results to the pipeline report.

        Parameters
        ----------
        pipeline_report : dict
            The pipeline report containing relevant results for the entire pipeline.
        """
        self._config = Config()
        self._pipeline_report = pipeline_report
        self._estimator = pipeline_report["estimator"]
        self._train_x = pipeline_report["train_x"]
        self._train_true_y = pipeline_report["train_true_y"]
        self._test_x = pipeline_report["test_x"]
        self._test_true_y = pipeline_report["test_true_y"]

    def run(self) -> dict:
        """
        Evaluates `estimator`'s performance on training and test datasets
        using various metrics and adds the results to the pipeline report.

        Returns
        -------
        dict
            The pipeline report with evaluation results added.
        """
        logger.info(f"Testing model: {type(self._estimator).__name__}")
        _avg = "weighted"

        # Compute train stats
        train_pred_y = self._estimator.predict(self._train_x)

        # Compute model evaluation metrics for train set
        train_precision = precision_score(
            self._train_true_y, train_pred_y, average=_avg, zero_division=np.nan
        )
        train_recall = recall_score(
            self._train_true_y, train_pred_y, average=_avg, zero_division=np.nan
        )

        # Compute testing stats
        test_pred_y = self._estimator.predict(self._test_x)

        # Compute model evaluation metrics for test set
        test_precision = precision_score(
            self._test_true_y, test_pred_y, average=_avg, zero_division=np.nan
        )

        test_recall = recall_score(
            self._test_true_y, test_pred_y, average=_avg, zero_division=np.nan
        )

        # Compute model accuracies on train and test using all selected scoring functions
        train_accuracies = {}
        test_accuracies = {}
        for func_name, func in ScoreFunctionSelector.getScoreFuncsModel().items():
            logger.info(f"Computing model accuracy using '{func_name}'")
            train_accuracies |= {
                func_name: func(self._estimator, self._train_x, self._train_true_y)
            }

            test_accuracies |= {
                func_name: func(self._estimator, self._test_x, self._test_true_y)
            }

        self._pipeline_report |= {
            "train_accuracies": train_accuracies,  # type: list[dict[str, float]]
            "train_precision": train_precision,  # type: float
            "train_recall": train_recall,  # type: float
            "test_accuracies": test_accuracies,  # type: list[dict[str, float]]
            "test_precision": test_precision,  # type: float
            "test_recall": test_recall,  # type: float
            "train_pred_y": train_pred_y,  # type: ndarray
            "test_pred_y": test_pred_y,  # type: ndarray
        }
        return self._pipeline_report
