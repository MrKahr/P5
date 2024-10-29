import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from copy import deepcopy
from typing import Any
from modules.config.config import Config
from modules.logging import logger
from modules.types import FittedEstimator


class ModelSummary:

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

    def _roundConvert(self, value: Any, digits: int = 3) -> str:
        if isinstance(value, int):
            return f"{value}"
        try:
            return f"{value:.{digits}f}"
        except TypeError as e:
            return f"{value}"

    def run(self) -> None:
        formatted = ""
        for k, v in deepcopy(self._model_report).items():
            if k == "feature_importances":
                continue
            if isinstance(v, dict):
                for tk, tv in v.items():
                    v[tk] = self._roundConvert(tv)
            formatted += f"\t{k}: {self._roundConvert(v)}\n"

        logger.info(f"Showing model report:\n{formatted}")

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
    def plotRocCurve(self) -> None:
        classifier = self._estimator
        test_X = self._test_x # Load from test dataset, input features and day
        test_y = self._test_true_y

        disp = RocCurveDisplay.from_estimator(classifier, test_X, test_y)
        disp.plot()

    # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html#sklearn.calibration.CalibrationDisplay
    def plotCalibrationDisplay(self) -> None:
        classifier = self._estimator
        test_X = self._test_x # Load from test dataset, input features and day
        test_y = self._test_true_y

        disp = CalibrationDisplay.from_estimator(classifier, test_X, test_y)
        disp.plot()
        plt.show()

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    def plotConfusionMatrix(self) -> None:
        y_true = self._test_true_y
        y_pred = self._estimator.predict(self._test_x) # Load predictions based on the test data

        confusion_matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)

        disp.plot()
        plt.show()


    # https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay
    # https://scikit-learn.org/stable/modules/partial_dependence.html
    def plotPartialDependence(self) -> None:
        classifier = self._estimator
        train_X = self._train_x
        features = None # TODO: Add features we want to inspect dependence between

        disp = PartialDependenceDisplay.from_estimator(classifier, train_X, features)
        disp.plot()
        plt.show()