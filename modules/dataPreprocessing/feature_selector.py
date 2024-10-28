from typing import Any, Callable, Literal, Union
import pandas as pd
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    chi2,
    f_classif,
    mutual_info_classif,
)
from numpy.typing import NDArray
from modules.config.config import Config
from modules.config.config_enums import FeatureScoreFunc, FeatureSelectionCriterion
from modules.logging import logger

# SECTION
# https://scikit-learn.org/stable/modules/feature_selection.html
# Tree-based: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py


# SECTION
# GRID SEARCH (hyperparameter tuning) {On any hyperparamter}

# Custom refit strategy of a grid search with cross-validation
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

# Running GridSearchCV using multiple evaluation metrics
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

# Statistical comparison of models:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py


# TODO: We need a method to generate random features completely independent from dataset for use in verification
class FeatureSelector:
    def __init__(self, train_x: pd.DataFrame, true_y: pd.Series) -> None:
        self._config = Config()
        self._parent_key = "FeatureSelection"
        self.df = None
        self._train_x = train_x
        self._true_y = true_y
        self._selected_features = None

    def __modeArgCompare(
        self,
    ) -> tuple[Literal["percentile", "k_best", "fpr", "fdr", "fwe"], int | float | str]:
        """Auxiliary function for GenericUnivariate select.
          It compares and selecting the right mode and ensuring that supplied args are type correctly

        Returns
        -------
        Pair containing mode name and mode arg

        Raises
        ------
        TypeError
            If supplied arg does not match mode.
        """

        # Get args associated with selection of mode
        parent_key = "GenericUnivariateSelectArgs"
        arg = self._config.getValue("param", parent_key)
        mode = self._config.getValue("mode", parent_key)

        # Use boolean to check whether a mode/param arg is a valid permutation
        isinteger = isinstance(arg, int)
        isnumeric = isinteger | isinstance(arg, float)

        # We need to check that mode and arg match
        match mode:
            case FeatureSelectionCriterion.PERCENTILE.name:
                if not isnumeric:
                    raise TypeError("percentiles must be specified as numeric")
                return ("percentile", arg)
            case FeatureSelectionCriterion.K_BEST.name:
                if not isinteger:
                    raise TypeError("k_best must be specified as numeric")
                return ("k_best", arg)
            case FeatureSelectionCriterion.FPR.name:
                if not isnumeric:
                    raise TypeError("fpr must be specified as numeric")
                return ("fpr", arg)
            case FeatureSelectionCriterion.FDR.name:
                if not isnumeric:
                    raise TypeError("fdr must be specified as numeric")
                return ("fdr", arg)
            case FeatureSelectionCriterion.FWE.name:
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

    def _chi2Independence(
        self, train_x: NDArray, true_y: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Compute chi-squared stats between each non-negative feature and class.
        This score can be used to select the features with the highest values for the
        test chi-squared statistic from `X`.

        Recall that the chi-square test measures dependence between stochastic variables, so using this
        function “weeds out” the features that are the most likely to be independent of class and therefore
        irrelevant for classification.

        Notes
        -----
        The chi-squared test should only be applied to non-negative features.

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

    def _fClassifIndependence(
        self, train_x: NDArray, true_y: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Compute the Analysis of Variance (ANOVA) F-value for the provided sample.

        Notes
        -----
        F-test estimate the degree of linear dependency between two random variables.

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

    def _mutualInfoClassif(
        self, train_x: NDArray, true_y: NDArray, **kwargs
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

    def genericUnivariateSelect(
        self,
        scoreFunc: Callable[[NDArray, NDArray], tuple[NDArray, NDArray] | NDArray],
        mode: Literal["percentile", "k_best", "fpr", "fdr", "fwe"],
        param: Union[int, float, str],
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

        Links
        -----
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html

        """
        # TODO: Plot info gained, see links
        # TODO: Try a new version of our custom score function as well
        logger.info(f"Running feature selection using mode={mode}, param={param}")
        selector = GenericUnivariateSelect(scoreFunc, mode=mode, param=param)
        transformed_x = selector.fit_transform(self._train_x, self._true_y)
        self._selected_features = selector.get_feature_names_out()
        self._train_x = pd.DataFrame(transformed_x, columns=self._selected_features)

    def varianceThreshold(self) -> Any:
        # See: https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        pass

    def checkOverfitting(self) -> Any:
        # NOTE: This appears to only work for tree-based models
        # See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
        pass

    # FIXME: Temporary solution until multi-score is implemented
    def getScoreFunc(
        self,
    ) -> Callable[[NDArray, NDArray], tuple[NDArray, NDArray] | NDArray]:
        score_funcs = self._config.getValue("score_functions", self._parent_key)
        if not isinstance(score_funcs, list):
            score_funcs = [score_funcs]

        selected_score_funcs = {}
        for score_func in score_funcs:
            if score_func == FeatureScoreFunc.CHI2.name:
                selected_score_funcs |= {
                    FeatureScoreFunc.CHI2.name.lower(): self._chi2Independence
                }
            elif score_func == FeatureScoreFunc.ANOVA_F.name:
                selected_score_funcs |= {
                    FeatureScoreFunc.ANOVA_F.name.lower(): self._fClassifIndependence
                }
            elif score_func == FeatureScoreFunc.MUTUAL_INFO_CLASSIFER.name:
                selected_score_funcs |= {
                    FeatureScoreFunc.MUTUAL_INFO_CLASSIFER.name.lower(): lambda train_x, true_y: self._mutualInfoClassif(
                        train_x=train_x,
                        true_y=true_y,
                        **self._config.getValue(
                            "MutualInfoClassifArgs", self._parent_key
                        ),
                    )
                }
            else:
                raise TypeError(
                    f"Invalid score function '{score_func}'. Expected one of {FeatureScoreFunc._member_names_}"
                )
        logger.info(
            f"Using feature select score functions: '{", ".join(selected_score_funcs.keys())}'"
        )
        return next(
            iter(selected_score_funcs.values())
        )  # FIXME: Temporary solution until multi-score is implemented

    def run(self) -> tuple[pd.DataFrame, pd.Series, NDArray]:
        """Runs all applicable feature selection methods.

        Returns
        -------
        pd.DataFrame
            The dataframe containing only select features.
        """
        if self._config.getValue("UseFeatureSelector"):
            if self._config.getValue("ComputeFeatureCorrelation", self._parent_key):
                self._computeFeatureCorrelation()
            if self._config.getValue("GenericUnivariateSelect", self._parent_key):
                self.genericUnivariateSelect(
                    self.getScoreFunc(),
                    *self.__modeArgCompare(),
                )
            if self._config.getValue("VarianceThreshold", self._parent_key):
                self.varianceThreshold()
            if self._config.getValue("checkOverfitting", self._parent_key):
                self.checkOverfitting()

        if self._selected_features is not None:
            size = len(self._selected_features)
            logger.info(
                f"Selected {size} feature{"s" if size != 1 else ""} as statistically important: {", ".join(self._selected_features)}"
            )
        else:
            logger.info("Using all features")
            self._selected_features = self._train_x.columns

        return self._train_x, self._true_y, self._selected_features
