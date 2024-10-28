from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.base import BaseEstimator


from copy import deepcopy
from typing import Any
from modules.logging import logger


class ModelSummary:

    def __init__(self, model_report: dict):
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
        classifier = None # TODO: Get the current classifier
        X_test, y_test = None # TODO: Load from test dataset, input features to X_test and day to y_test

        disp = RocCurveDisplay.from_estimator(classifier, X_test, y_test)
        disp.plot()

    # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html#sklearn.calibration.CalibrationDisplay
    def plotCalibrationDisplay(self) -> None:
        classifier = None # TODO: Get the current classifier
        X_test, y_test = None # TODO: Load from test dataset, input features to X_test and day to y_test

        disp = CalibrationDisplay.from_estimator(classifier, X_test, y_test)
        disp.plot()
        plt.show()

    # https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay
    # https://scikit-learn.org/stable/modules/partial_dependence.html
    def plotPartialDependence(self) -> None:
        classifier = None # TODO: Get the current classifier
        X, features = None # TODO: Load input features values to X and feature labels to features

        disp = PartialDependenceDisplay.from_estimator(classifier, X, features)
        disp.plot()
        plt.show()

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    def plotConfusionMatrix(self) -> None:
        y_true = [2, 0, 2, 2, 0, 1] # TODO: Get the true and predicted values from test data
        y_pred = [0, 0, 2, 2, 0, 2]

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()
        plt.show()

summarizer = ModelSummary()
summarizer.plotConfusionMatrix()