from numpy import ndarray
from numpy.typing import NDArray
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from modules.logging import logger


class ModelScoreFunctions:
    @classmethod
    def threshold(
        cls,
        true_y: pd.Series | NDArray,
        pred_y: NDArray,
        threshold: int,
    ) -> float:
        """
        Scoring function based on a specified `threshold`.

        E.g., is the predicted y-value above or below the true y-value by 20?

        Parameters
        ----------
        true_y : pd.Series | NDArray
            Target feature, i.e., "Dag".

        pred_y : NDArray
            Predicted target feature, i.e., "Dag".

        threshold : int
            The threshold which `pred_y` and `true_y` are compared against.

        Returns
        -------
        float
            The model's accuracy according to this score function.
        """
        score = 0
        if not isinstance(true_y, ndarray):
            true_y = true_y.to_numpy(copy=True)
        for i, prediction in enumerate(pred_y):
            if true_y[i] >= threshold and prediction >= threshold:
                score += 1
            elif true_y[i] < threshold and prediction < threshold:
                score += 1
        return float(score / len(pred_y))

    @classmethod
    def distance(cls, true_y: pd.Series | NDArray, pred_y: NDArray) -> float:
        """
        Scoring function based on the distance `pred_y` is from `true_y`.

        E.g., `true_y` = 5 , `pred_y` = 7; distance = 2.

        Parameters
        ----------
        true_y : pd.Series | NDArray
            Target feature, i.e., "Dag".

        pred_y : NDArray
            Predicted target feature, i.e., "Dag".

        Returns
        -------
        float
            The model's accuracy according to this score function.
        """
        score = 0
        if not isinstance(true_y, ndarray):
            true_y = true_y.to_numpy(copy=True)
        max_y = true_y[true_y.argmax()]
        for i, prediction in enumerate(pred_y):
            score += 1 - abs(prediction - true_y[i]) / max_y
        return float(score / len(pred_y))


class FeatureSelectScoreFunctions:

    @classmethod
    def chi2Independence(
        self, train_x: NDArray, true_y: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Compute chi-squared stats between each non-negative feature and class.
        This score can be used to select the features with the highest values for the
        test chi-squared statistic from the training data.

        Recall that the chi-square test measures dependence between stochastic variables, so using this
        function “weeds out” the features that are the most likely to be independent of class and therefore
        irrelevant for classification.

        Notes
        -----
        The chi-squared test should only be applied to non-negative features.

        Parameters
        ----------
        train_x : NDArray
            Training feature(s).

        target_y : NDArray
            Target feature, i.e., "Dag".

        Returns
        -------
        tuple[NDArray, NDArray]
            [0]: Chi-squared statistics for the input data.
            [1]: P-values for the input data.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
        """
        logger.info("Computing chi-squared statistics")
        chi2_stats, p_values = chi2(train_x, true_y)
        return chi2_stats, p_values

    @classmethod
    def fClassifIndependence(
        self, train_x: NDArray, true_y: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Compute the Analysis of Variance (ANOVA) F-value for the provided sample.

        Notes
        -----
        F-test estimate the degree of linear dependency between two random variables.

        Parameters
        ----------
        train_x : NDArray
            Training feature(s).

        target_y : NDArray
            Target feature, i.e., "Dag".

        Returns
        -------
        tuple[NDArray, NDArray]
            [0]: F-statistics for the input data.
            [1]: P-values for the input data.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
        - https://en.wikipedia.org/wiki/F-test

        """
        logger.info("Computing ANOVA F-statistics")
        f_stats, p_values = f_classif(train_x, true_y)
        return f_stats, p_values

    @classmethod
    def mutualInfoClassif(self, train_x: NDArray, true_y: NDArray, **kwargs) -> NDArray:
        """
        Estimate mutual information for a discrete target variable.
        Mutual information (MI) between two random variables is a non-negative value,
        which measures the dependency between the variables. It is equal to zero if and only if
        two random variables are independent, and higher values mean higher dependency.

        The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances.

        Notes
        -----
        - The term “discrete features” is used instead of naming them “categorical”, because it describes the essence more accurately.
        - Mutual information methods can capture any kind of statistical dependency, but being nonparametric, they require more samples for accurate estimation.
        - Also note, that treating a continuous variable as discrete and vice versa will usually give incorrect results, so be attentive about that.

        Parameters
        ----------
        train_x : NDArray
            Training feature(s).

        target_y : NDArray
            Target feature, i.e., "Dag".

        **kwargs : dict
            Additional parameters defined in the config.

        Returns
        -------
        NDArray
            Estimated mutual information between each feature and the target in nat units.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
        """
        logger.info("Computing estimates of mutual information")
        return mutual_info_classif(
            train_x,
            true_y,
            **kwargs,
        )
