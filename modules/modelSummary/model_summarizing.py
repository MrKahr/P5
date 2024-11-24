from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
from typing import Any
from itertools import cycle

import numpy as np
from numpy.typing import ArrayLike, NDArray

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.naive_bayes import LabelBinarizer
from sklearn.tree import plot_tree
from sklearn.utils._param_validation import InvalidParameterError

from modules.config.config import Config
from modules.logging import logger


# TODO: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report


class ModelSummary:

    def __init__(self, model_report: dict):
        self._config = Config()
        self._model_report = model_report  # Values added in model_test run
        self._write_fig = self._config.getValue("write_figure_to_disk")
        self._model_name = type(self._model_report["estimator"]).__name__

    def _writeFigure(self, figure_name: str) -> None:
        figures = "figures"
        os.makedirs(figures, exist_ok=True)
        plt.savefig(
            Path(
                Path.cwd(),
                figures,
                f"{figure_name}.{self._model_name}.{datetime.now().strftime("%Y-%m-%d")}.png",
            )
        )

    def _roundConvert(self, value: Any, digits: int = 3) -> str:
        if isinstance(value, int):
            return f"{value}"
        try:
            return f"{value:.{digits}f}"
        except TypeError:
            return f"{value}"

    def _printModelReport(self):
        """print results for model evaluation from model_report"""
        formatted = ""
        for k, v in deepcopy(self._model_report).items():
            if k == "feature_importances":
                continue
            if k == "train_pred_y":
                break  # Avoids printing extra stuff used in plotting
            if isinstance(v, dict):
                for tk, tv in v.items():
                    v[tk] = self._roundConvert(tv)
            formatted += f"\t{k}: {self._roundConvert(v)}\n"

        logger.info(f"Showing model report:\n{formatted}")

    def _computeAverages(
        self, y_onehot_test: NDArray, y_score: ArrayLike, n_classes: int
    ) -> dict:
        """
        Compute the micro and macro average of Roc curves for plotRocCurve.

        Parameters
        ----------
        y_onehot_test : NDArray
            Days from test dataset after one-hot-encoding.

        y_score : np.ArrayLike
            Probability estimates the used classifier on the test dataset.

        n_classes : int
            Number of unique categories the classifier predicts for.

        Returns
        -------
        dict
            fpr (false positive rate), tpr (true positive rate), and roc_auc (ROC area under curve)

        Links
        -----
        - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#roc-curve-using-micro-averaged-ovr
        - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#roc-curve-using-the-ovr-macro-average
        """
        # fpr (false positive rate), tpr (true positive rate), and roc_auc (ROC area under curve)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], pos_label = roc_curve(
            y_onehot_test.ravel(), y_score.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i in range(n_classes):
            # [:, i] = Numpy indexing for multi-dimensional arrays
            fpr[i], tpr[i], pos_label = roc_curve(y_onehot_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)

        # Compute macro tpr
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return fpr, tpr, roc_auc

    def _plotRocCurve(self) -> None:
        """
        Plot ROC (Receiver Operating Characteristic) Curve using
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html

        Adapted to work with multiple categories using OvR strategy based on
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#one-vs-one-multiclass-roc

        `The One-vs-the-Rest (OvR) multiclass strategy, also known as one-vs-all, consists in computing a
        ROC curve per each of the n_classes. In each step, a given class is regarded as the positive class
        and the remaining classes are regarded as the negative class as a bulk.`
        """

        # Load model results from model report
        train_true_y = self._model_report["train_true_y"]
        test_true_y = self._model_report["test_true_y"]
        y_score = self._model_report["estimator"].predict_proba(
            self._model_report["test_x"]
        )

        # Get the different days trained on
        target_names = np.unique(train_true_y)
        n_classes = len(target_names)

        # Split categorical classification into multiple binary classifications (one-vs-all/one-vs-rest)
        label_binarizer = LabelBinarizer().fit(train_true_y)
        y_onehot_test = label_binarizer.transform(test_true_y)
        y_onehot_test.shape  # (n_samples, n_classes)

        # Store the fpr (false positive rate), tpr (true positive rate), and roc_auc (ROC area under curve) for micro and macro averaging strategies
        fpr, tpr, roc_auc = self._computeAverages(y_onehot_test, y_score, n_classes)

        # Plot every curve as subplots
        figure, axes = plt.subplots(figsize=(10, 10))

        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(
            mcolors.XKCD_COLORS
        )  # Long list of colors from https://matplotlib.org/stable/gallery/color/named_colors.html
        for class_id, color in zip(range(n_classes), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_score[:, class_id],
                name=f"ROC curve for {target_names[class_id]}",
                color=color,
                ax=axes,
                plot_chance_level=(class_id == 2),
            )

        axes.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"({self._model_name})\nExtension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
        )
        if self._write_fig:
            self._writeFigure("roc_auc")
        else:
            plt.show(block=False)

    def _plotConfusionMatrix(self) -> None:
        """
        Plot Confusion Matrix using
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        """
        disp = ConfusionMatrixDisplay.from_estimator(
            self._model_report["estimator"],
            self._model_report["test_x"],
            self._model_report["test_true_y"],
        )
        disp.plot()

        if self._write_fig:
            self._writeFigure("confusion_matrix")
        else:
            plt.show(block=False)

    def _plotTree(self) -> None:
        plt.figure(dpi=1200)
        tree = self._model_report["estimator"]
        try:
            plot_tree(
                tree,
                filled=True,
                fontsize=1,
                rounded=True,
                feature_names=self._model_report["feature_names_in"],
            )
        except InvalidParameterError:
            return
        plt.title(
            f"{type(tree).__name__} trained on {self._model_report["feature_count"]} features"
        )
        if self._write_fig:
            self._writeFigure("tree")
        else:
            plt.show(block=False)

    def run(self) -> None:
        if self._config.getValue("print_model_report"):
            self._printModelReport()
        if self._config.getValue("plot_confusion_matrix"):
            self._plotConfusionMatrix()
        if self._config.getValue("plot_roc_curves"):
            self._plotRocCurve()
        if self._config.getValue("plot_tree"):
            self._plotTree()

        if not self._write_fig:
            input("Press enter to close all figures...")
