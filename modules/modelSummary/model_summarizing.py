from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
from typing import Any
from itertools import cycle

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.naive_bayes import LabelBinarizer
from sklearn.tree import plot_tree

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import Model
from modules.config.utils.setup_config import SetupConfig
from modules.logging import logger


# TODO: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report


class ModelSummary:
    def __init__(self, pipeline_report: dict):
        self._config = PipelineConfig()
        self._write_fig = self._config.getValue("write_figure_to_disk")
        self._model_name = type(pipeline_report["estimator"]).__name__
        self._pipeline_report = pipeline_report

    def _showFigure(self, figure: Figure, figure_filename: str) -> None:
        if self._write_fig:
            figure_path = Path(SetupConfig().figures_dir, self._model_name)
            config_filename = os.path.splitext(
                os.path.split(self._config.getConfigPath())[1]
            )[0]
            os.makedirs(figure_path, exist_ok=True)
            plt.savefig(
                Path(
                    figure_path,
                    f"{figure_filename}_{config_filename}_{self._model_name}.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png",
                )
            )
            plt.close(figure)
        else:
            plt.show(block=False)

    def _printModelReport(self):
        """Print results for model evaluation from pipe_line_report"""

        def roundConvert(value: Any, digits: int = 3) -> str:
            if isinstance(value, int):
                return f"{value}"
            try:
                return f"{value:.{digits}f}"
            except (ValueError, TypeError):
                return f"{value}"

        formatted = "{\n"
        for k, v in deepcopy(self._pipeline_report).items():
            if k in [
                "feature_importances",
                "train_pred_y",
                "test_pred_y",
            ] or isinstance(v, (pd.DataFrame, pd.Series)):
                continue
            elif isinstance(v, (list, np.ndarray)):
                formatted += f"\t{k}: [\n\t    {"\n\t    ".join([f"'{roundConvert(item)}'," for item in v])}\n\t]\n"
                continue
            elif isinstance(v, dict):
                for tk, tv in v.items():
                    v[tk] = roundConvert(tv)
            formatted += f"\t{k}: {roundConvert(v)}\n"
        formatted += "}\n"
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
        train_y = self._pipeline_report["train_y"]
        test_y = self._pipeline_report["test_y"]
        y_score = self._pipeline_report["estimator"].predict_proba(
            self._pipeline_report["test_x"]
        )

        # Get the different days trained on
        target_names = np.unique(train_y)
        n_classes = len(target_names)

        # Split categorical classification into multiple binary classifications (one-vs-all/one-vs-rest)
        label_binarizer = LabelBinarizer().fit(train_y)
        y_onehot_test = label_binarizer.transform(test_y)
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
        self._showFigure(figure, "roc_auc")

    def _plotConfusionMatrix(self) -> None:
        """
        Plot Confusion Matrix using
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        """
        disp = ConfusionMatrixDisplay.from_estimator(
            self._pipeline_report["estimator"],
            self._pipeline_report["test_x"],
            self._pipeline_report["test_y"],
        )
        disp.plot()
        self._showFigure(disp.figure_, "confusion_matrix")

    # FIXME: Half-baked implementation!
    def _plotDecisionBoundary(self) -> None:
        raise NotImplementedError("FIXME: Half-baked implementation!")
        x = self._pipeline_report["train_x"]
        print(x)
        dbd = DecisionBoundaryDisplay.from_estimator(
            self._pipeline_report["estimator"], x
        )
        dbd.ax_.scatter(
            x[:, 0], x[:, 1], c=self._pipeline_report["train_y"], edgecolor="k"
        )
        self._showFigure("decision_boundary")

    def _plotTree(self) -> None:
        selected_model = self._config.getValue("model", "ModelSelection")
        if selected_model in [Model.DECISION_TREE.name]:
            figure = plt.figure(dpi=1200)
            tree = self._pipeline_report["estimator"]
            plot_tree(
                tree,
                filled=True,
                fontsize=1,
                rounded=True,
                feature_names=tree.feature_names_in_,
            )
            plt.title(
                f"{type(tree).__name__} trained on {self._pipeline_report["feature_count"]} features"
            )
            self._showFigure(figure, "tree")

    def _plotFeatureImportance(self) -> None:
        """
        Feature importance groups (threshold, distance, accuracy, balanced accuracy) are calculated by summing the importances_mean
        of each feature for these metrics. The features are then grouped, sorted by their sums, and plotted in the determined order.

        Variables:
            - result = List with [threshold,distance,accuracy,balanced_accuracy]
                Note: Getting permutation feature importances from model training containing threshold, distance, accuracy and
                balanced_accuracy which all consist of importances_mean, importances_std and importances for each feature.
            - feature_names = List with [names of features used,,...].
                Note: we load is because result doesn't save feature names(only values)
            - model = String being the model used.
                Note: loaded because we want the plots show which model was used
            - feature_groups = List with [groupFeaturen_name(threshold.importances_mean,distance.importances_mean,accuracy.importances_mean,balanced_accuracy.importances_mean),(),(),...]
                Note: this list is used for calculating the sum of the means.
            - overall_means = List of tuples [(sum of (threshold,distance,accuracy,balanced_accuracy), feature name),(,)(,)...].
                Note: we have a tuple with sum and feature name for each feature_group which then is sorted where we want to obtain
                the feature names to order the feature groups in the plot. (we use sum value to get the sorted order)
            - sorted_group_order = List with [feature name,,,...] Note: this list is needed for plotting the feature groups in the
                sorted order.

        Links
        -----
        - https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        - https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        """
        result = self._pipeline_report["feature_importances"]
        feature_names = self._pipeline_report["feature_names_in"]
        feature_groups = {f"group{name}": [] for name in feature_names}
        overall_means = []
        sorted_group_order = []

        for i, x in enumerate(result):
            labeled_feature_values = pd.Series(
                result[x].importances_mean, index=feature_names
            )  # Label the means with their feature name

            for j, feature in enumerate(labeled_feature_values):
                feature_groups[f"group{labeled_feature_values.index[j]}"].append(
                    feature
                )  # Add the feature mean to corresponding feature group

                if i == (
                    len(result) - 1
                ):  # Calculate sum of all feature groups and sort the groups
                    # if you want to get the mean of means then change sum() to mean()
                    overall_means.append(
                        (
                            sum(
                                feature_groups[
                                    f"group{labeled_feature_values.index[j]}"
                                ]
                            ),
                            feature_names[j],
                        )
                    )
                    overall_means.sort()  # Should only sort on sum values [0]
                    sorted_group_order = [
                        n[1] for n in overall_means
                    ]  # Should only collect feature names [1]

        fig, ax = plt.subplots(figsize=(20, 10), layout="constrained")
        # Variables are used to position the feature groupings(threshold,distance,accuracy,balanced_accuracy) in the plot.
        positions = np.arange(len(feature_names)) * 10
        width = 2
        multiplier = 0

        for y in result:
            offset = width * multiplier
            feature_importances = pd.Series(
                result[y].importances_mean, index=feature_names
            )
            sorted_feature_importances = feature_importances[
                sorted_group_order
            ]  # To not sort change sorted_group_order to feature_names

            feature_plot = ax.barh(
                positions + offset,
                sorted_feature_importances,
                width,
                xerr=result[y].importances_std,
                label=y,
            )
            ax.bar_label(feature_plot, padding=1)
            multiplier += 1

        n_repeats = self._config.getValue("n_repeats", "PermutationFeatureImportance")
        ax.set_xlabel("Mean accuracy")
        ax.set_ylabel("Features")
        ax.set_title(
            f"Feature Importances with Standard Deviation\n{self._model_name} ({n_repeats} repeats)"
        )
        ax.legend(loc="best")
        ax.set_yticks(positions + (width * (len(result) - 1) / 2))
        ax.set_yticklabels(
            sorted_group_order
        )  # To not sort change sorted_group_order to feature_names

        self._showFigure(fig, "feature importance")

    def run(self) -> None:
        if self._config.getValue("print_model_report"):
            self._printModelReport()
        if self._config.getValue("plot_confusion_matrix"):
            self._plotConfusionMatrix()
        if self._config.getValue("plot_roc_curves"):
            self._plotRocCurve()
        if self._config.getValue("plot_decision_boundary"):
            self._plotDecisionBoundary()
        if self._config.getValue("plot_tree"):
            self._plotTree()
        if self._config.getValue("plot_feature_importance"):
            self._plotFeatureImportance()

        if not self._write_fig:
            input("Press enter to close all figures...")
