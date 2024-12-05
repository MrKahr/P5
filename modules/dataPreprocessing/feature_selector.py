from typing import Literal, Union
from sklearn.feature_selection import GenericUnivariateSelect
from numpy.typing import NDArray
import pandas as pd

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import FeatureSelectionCriterion
from modules.logging import logger
from modules.scoreFunctions.score_function_selector import ScoreFunctionSelector
from modules.tools.types import FeatureSelectScoreCallable


class FeatureSelector:
    def __init__(self, train_x: pd.DataFrame, true_y: pd.Series) -> None:
        """
        Using algorithms from statistics, figure out which features to use for model training.

        Parameters
        ----------
        train_x : pd.DataFrame
            Training feature(s).

        true_y : pd.Series
            Target feature, i.e., "Dag".
        """
        self._train_x = train_x
        self._true_y = true_y
        self._config = PipelineConfig()
        self._parent_key = "FeatureSelection"
        self.df = None
        self._selected_features = None

    def __modeArgCompare(
        self,
    ) -> tuple[Literal["percentile", "k_best", "fpr", "fdr", "fwe"], int | float | str]:
        """
        Auxiliary function for `GenericUnivariateSelect`.\n
        It ensures that the config arguments `mode` and `param` are a valid combination.

        Returns
        -------
        tuple[Literal["percentile", "k_best", "fpr", "fdr", "fwe"], int | float | str]
            [0]: `mode`.
            [1]: `param`.

        Raises
        ------
        TypeError
            If the config arguments `mode` and `param` are an invalid combination.

        ValueError
            If `mode` is invalid.
        """

        # Get `mode` and `paran` from the config
        parent_key = "GenericUnivariateSelectArgs"
        arg = self._config.getValue("param", parent_key)
        mode = self._config.getValue("mode", parent_key)

        # Check whether a mode/param arg is a valid permutation
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
                raise ValueError(
                    f"Invalid mode '{mode}' selected. Expected one of {FeatureSelectionCriterion._member_names_}"
                )

    def genericUnivariateSelect(
        self,
        scoreFunc: FeatureSelectScoreCallable,
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
        scoreFunc : FeatureSelectScoreCallable
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
        logger.info(f"Running feature selection using mode={mode}, param={param}")
        selector = GenericUnivariateSelect(scoreFunc, mode=mode, param=param)
        transformed_x = selector.fit_transform(self._train_x, self._true_y)
        self._selected_features = selector.get_feature_names_out()
        self._train_x = pd.DataFrame(transformed_x, columns=self._selected_features)

    def run(self) -> tuple[pd.DataFrame, pd.Series, NDArray]:
        """
        Runs all applicable feature selection methods.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, NDArray]
            [0]: Selected training feature(s).
            [1]: Target feature, i.e., "Dag".
            [2]: Selected feature labels.
        """
        if self._config.getValue("UseFeatureSelector"):
            if self._config.getValue("ComputeFeatureCorrelation", self._parent_key):
                self._computeFeatureCorrelation()
            if self._config.getValue("GenericUnivariateSelect", self._parent_key):
                self.genericUnivariateSelect(
                    ScoreFunctionSelector.getScoreFuncFeatureSelect(),
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
            self._selected_features = self._train_x.columns
            logger.info(
                f"Skipping feature selection ({len(self._selected_features)} features present)"
            )

        return self._train_x, self._true_y
