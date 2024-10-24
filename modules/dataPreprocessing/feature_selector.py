from typing import Any, Callable, Literal, Optional, Self, Union
import pandas as pd
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    chi2,
    f_classif,
    mutual_info_classif,
)
from numpy.typing import NDArray
from sklearn.inspection import permutation_importance
from modules.config.config import Config
from modules.config.config_enums import FeatureSelectionCriterion
from modules.logging import logger
from modules.modelTesting.model_testing import ModelTester

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
    def __init__(self, x_train: NDArray, y_train: NDArray) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def __modeArgCompare(
        self, featureSelectionCriterion: FeatureSelectionCriterion, config: Config
    ) -> tuple[Literal["percentile", "k_best", "fpr", "fdr", "fwe"], int | float | str]:
        """Auxiliary function for GenericUnivariate select.
          It compares and selecting the right mode and ensuring that supplied args are type correctly

        Returns
        -------
        Pair containing mode name and mode arg

        Raises
        ------
        TypeError
            Error raised if supplied arg does not match mode.
        """

        # Get args associated with selection of mode
        arg = config.getValue("param", "GenericUnivariateSelectArgs")

        # Use boolean to check whether a mode/param arg is a valid permutation
        isnumeric = isinstance(arg, int) | isinstance(arg, float)
        isinteger = isinstance(arg, int)

        # We need to check that mode and arg match
        match featureSelectionCriterion:
            case FeatureSelectionCriterion.PERCENTILE:
                if not isnumeric:
                    raise TypeError("percentiles must be specified as numeric")
                return ("percentile", arg)
            case FeatureSelectionCriterion.K_BEST:
                if not isinteger:
                    raise TypeError("k_best must be specified as numeric")
                return ("k_best", arg)
            case FeatureSelectionCriterion.FPR:
                if not isnumeric:
                    raise TypeError("fpr must be specified as numeric")
                return ("fpr", arg)
            case FeatureSelectionCriterion.FDR:
                if not isnumeric:
                    raise TypeError("fdr must be specified as numeric")
                return ("fdr", arg)
            case FeatureSelectionCriterion.FWE:
                if not isnumeric:
                    raise TypeError("fwe must be specified as numeric")
                return ("fwe", arg)
            case _:
                logger.warning(
                    "Assuming parameter: 'all' specified for generic univariate select"
                )
                return ("all", arg)

    def _computeFeatureCorrelation(self) -> Any:
        # Using Spearman rank-order correlations from SciPy
        # See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features
        pass

    def _chi2Independence(self) -> tuple[NDArray, NDArray]:
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
        chi2_stats, p_values = chi2(self.x_train, self.y_train)
        return chi2_stats, p_values

    def _fClassifIndependence(self) -> tuple[NDArray, NDArray]:
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
        f_stats, p_values = f_classif(self.x_train, self.y_train)
        return f_stats, p_values

    def _mutualInfoClassif(
        self,
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
            self.x_train,
            self.y_train,
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            copy=copy,
            random_state=random_state,
        )
        return mi

    def genericUnivariateSelect(
        self,
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
            else [f"Feature {i}" for i in range(self.x_train.shape[1])]
        )
        selector = GenericUnivariateSelect(scoreFunc, mode=mode, param=param)
        new_training_data = selector.fit_transform(self.x_train, self.y_train)
        remaining_columns = selector.get_feature_names_out(old_columns)
        logger.info(
            f"Selected {len(remaining_columns)} features as important: {remaining_columns}"
        )
        return new_training_data

    def varianceThreshold(self) -> Any:
        # See: https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        pass

    def checkOverfitting(self) -> Any:
        # NOTE: This appears to only work for tree-based models
        # See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
        pass

    def run(self, modelTester: ModelTester) -> None:
        """Runs all applicable features selection methods
        #NOTE - Currently needs a scoring function from model tester

        Parameters
        ----------
        modelTester : ModelTester
            Provides the score function from among its attributes
        """
        config = Config()
        if config.getValue("ComputeFeatureCorrelation"):
            self._computeFeatureCorrelation()
        if config.getValue("TestChi2Independence"):
            self._chi2Independence()
        if config.getValue("TestfClassifIndependence"):
            self._fClassifIndependence()
        if config.getValue("MutualInfoClassif"):
            self._mutualInfoClassif(
                **self.config.getValue("MutualInfoClassifArgs"),
            )
        if config.getValue("GenericUnivariateSelect"):
            self.genericUnivariateSelect(
                modelTester.getScoreFunc(),  # TODO - Placement is not idea as it requires initialization of model trainer
                **self.__modeArgCompare(
                    config.getValue("mode", "GenericUnivariateSelectArgs"), config
                ),
                x_labels=None,
            )
        if config.getValue("VarianceThreshold"):
            self.varianceThreshold()
        if config.getValue("checkOverfitting"):
            self.checkOverfitting()
        if config.getValue("recursiveFeatureValidation"):
            self.recursiveFeatureValidation()
        if config.getValue("recursiveFeatureValidationWithCrossValidation"):
            self.recursiveFeatureValidationWithCrossValidation()
