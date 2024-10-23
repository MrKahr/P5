from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.base import BaseEstimator


class ModelSummary:
    def __init__(self, training_df: DataFrame, test_df: DataFrame, model: BaseEstimator):
        """Load values to be used in summarizing model results

        Parameters
        ----------
        training_df : DataFrame
            The dataset trained on
        test_df : DataFrame
            The dataset for testing
        model : BaseEstimator
            The machine learning model used, e.g. RandomForestClassifier
        """
        self.training_df = training_df
        self.test_df = test_df
        self.model = model

    @classmethod
    def run(self) -> None:
        print(f"{__name__}is run")

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