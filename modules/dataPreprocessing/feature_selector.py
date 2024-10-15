from typing import Any, Callable, Literal, Optional, Union
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    chi2,
    f_classif,
    mutual_info_classif,
)
from numpy.typing import NDArray
from sklearn.inspection import permutation_importance

from modules.logging import logger

# SECTION
# https://scikit-learn.org/stable/modules/feature_selection.html
# Tree-based: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py


# SECTION
# GRID SEARCH (hyperparameter tuning) {On any hyperparamter}

# How-to grid search
# https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search

# Custom refit strategy of a grid search with cross-validation
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

# Running GridSearchCV using multiple evaluation metrics
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

# Statistical comparison of models:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py

# SECTION
# RECURSIVE FEATURE ELIMINATION WITH CROSS-VALIDATION (hyperparameter tuning) {On cross-validation hyperparameters}

# How-to RFECV
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html


# TODO: We need a method to generate random features completely independent from dataset for use in verification
class FeatureSelector:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def _computeFeatureCorrelation(self) -> Any:
        # Using Spearman rank-order correlations from SciPy
        # See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features
        pass

    def _chi2Independence(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """
        Compute chi-squared stats between each non-negative feature and class.
        This score can be used to select the features with the highest values for the
        test chi-squared statistic from X.

        Recall that the chi-square test measures dependence between stochastic variables, so using this
        function “weeds out” the features that are the most likely to be independent of class and therefore
        irrelevant for classification.

        Notes
        -----
        The chi-squared test should only be applied to non-negative features.

        Parameters
        ----------
        X : NDArray
            Training data.

        y : NDArray
            Target data.

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
        chi2_stats, p_values = chi2(X, y)
        return chi2_stats, p_values

    def _fClassifIndependence(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """
        Compute the Analysis of Variance (ANOVA) F-value for the provided sample.

        Notes
        -----
        F-test estimate the degree of linear dependency between two random variables.

        Parameters
        ----------
        X : NDArray
            Training data.

        y : NDArray
            Target data.

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
        f_stats, p_values = f_classif(X, y)
        return f_stats, p_values

    def _mutualInfoClassif(
        self,
        X: NDArray,
        y: NDArray,
        discrete_features: Union[str, bool, NDArray] = True,
        n_neighbors: int = 3,
        copy: bool = True,
        random_state: int = 12,
    ) -> NDArray:
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
        X : NDArray
            Training data.

        y : NDArray
            Target data.

        discrete_features : str | bool | NDArray
            If bool, then determines whether to consider all features discrete
            or continuous. If array, then it should be either a boolean mask
            with shape (n_features,) or array with indices of discrete features.
            If 'auto', it is assigned to False for dense `X` and to True for
            sparse `X`.

        n_neighbors : int, optional
            Number of neighbors to use for MI estimation for continuous variables.
            Higher values reduce variance of the estimation, but could introduce a bias.
            By default 3.

        copy : bool, optional
            Whether to make a copy of the given data.
            If set to False, the initial data will be overwritten.
            By default True.

        random_state : int, optional
            Determines random number generation for adding small noise to continuous
            variables in order to remove repeated values. Pass an int for reproducible
            results across multiple function calls.
            By default 12.

        Returns
        -------
        NDArray
            Estimated mutual information between each feature and the target in nat units.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
        """
        logger.info("Computing estimates of mutual information")
        mi = mutual_info_classif(
            X,
            y,
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            copy=copy,
            random_state=random_state,
        )
        return mi

    def genericUnivariateSelect(
        self,
        X: NDArray,
        y: NDArray,
        scoreFunc: Callable[[NDArray, NDArray], tuple[NDArray, NDArray] | NDArray],
        mode: Literal["percentile", "k_best", "fpr", "fdr", "fwe"],
        param: Union[int, float, str],
        x_labels: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Univariate feature selector with configurable strategy.\n
        This allows to select the best univariate selection strategy with hyper-parameter search estimator.

        Notes
        -----
        In inductive learning, where the goal is to learn a generalized model that can be applied to new data,
        users should be careful not to apply fit_transform to the entirety of a dataset (i.e. training and test data together)
        before further modelling, as this results in data leakage.

        Parameters
        ----------
        X : NDArray
            Training data.

        y : NDArray
            Target data.

        scoreFunc : Callable
            Function taking two arrays `X` and `y`, and returning a pair of arrays (scores, pvalues) or a single array with scores.
            This could for instance be: 'chi2', 'f_classif', or 'mutual_info_classif'.

        mode : Literal["percentile", "k_best", "fpr", "fdr", "fwe"]
            Feature selection mode:
                percentile
                    Removes all but a user-specified highest scoring percentage of features.
                k_best
                    Removes all but the `k` highest scoring features.
                fpr
                    Select features based on a False Positive Rate test.
                fdr
                    Select features based on an estimated False Discovery Rate.
                fwe
                    Select features based on Family-Wise Error rate.

        param : Union[int, float, str]
            Parameter of the corresponding mode:
                percentile : int
                    Percent of features to keep.
                k_best : int | Literal["all"]
                    Number of top features to select. The "all" option bypasses selection, for use in a parameter search.
                fpr : float
                    Features with p-values less than `alpha` are selected.
                fdr : float
                    The highest uncorrected p-value for features to keep.
                    Features with p-values less than `alpha` are selected.
                fwe :
                    The highest uncorrected p-value for features to keep.
                    Features with p-values less than `alpha` are selected.

        Returns
        -------
        NDArray
            A new `X` array with only select features remaining.

        Links
        -----
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html

        """
        # TODO: Plot info gained, see links
        # TODO: Try a new version of our custom score function as well
        logger.info(f"Running feature selection using ({mode}, {param})")
        old_columns = (
            x_labels
            if len(x_labels) > 0
            else [f"Feature {i}" for i in range(X.shape[1])]
        )
        selector = GenericUnivariateSelect(scoreFunc, mode=mode, param=param)
        new_training_data = selector.fit_transform(X, y)
        remaining_columns = selector.get_feature_names_out(old_columns)
        logger.info(
            f"Selected {len(remaining_columns)} features as important: {remaining_columns}"
        )
        return new_training_data

    def varianceThreshold(self) -> Any:
        # See: https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        pass

    def permutationFeatureImportance(
        self,
        fitted_estimator: Any,
        X: NDArray,
        y: NDArray,
        scoring: Union[list[str], Callable],
        n_repeats: int,
        n_jobs: int = -1,
        random_state: int = 23,
    ) -> Any:
        """
        Permutation feature importance is a model inspection technique
        that measures the contribution of each feature to a fitted model's
        statistical performance on a given tabular dataset. This technique is
        particularly useful for non-linear or opaque estimators, and involves
        randomly shuffling the values of a single feature and observing the resulting
        degradation of the model's score. By breaking the relationship between the
        feature and the target, we determine how much the model relies on such particular feature.

        Notes
        -----
        Permutation importance does not reflect the intrinsic predictive value of a
        feature by itself but how important this feature is for a particular model.

        Parameters
        ----------
        fitted_estimator : Any
            A fitted estimator.

        X : NDArray
            Training data.

        y : NDArray
            Target data.

        scoring : Union[list[str], Callable]
            The score method used to measure feature importance.
            To use a preset score method (i.e. `list[str]`), please see:\n
            https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values\n
            for a table of all score methods.

        n_repeats : int
            Number of times to permute a feature.

        n_jobs : int, optional
            CPU cores used (`-1` means ALL).
            By default -1.

        random_state : int, optional
            Pseudo-random number generator to control the permutations of each feature.
            Pass an int to get reproducible results across function calls.
            By default 23.

        Returns
        -------
        Any
            _description_

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
        - https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance

        """
        # TODO: Implement plots from: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        # TODO: Verify that the model is better than a RNG: https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html#sphx-glr-auto-examples-model-selection-plot-permutation-tests-for-classification-py
        result = permutation_importance(
            fitted_estimator,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def checkOverfitting(self) -> Any:
        # NOTE: This appears to only work for tree-based models
        # See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
        pass

    def recursiveFeatureValidation(self) -> Any:
        # See: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py
        pass

    def recursiveFeatureValidationWithCrossValidation(self) -> Any:
        # See: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
        pass
