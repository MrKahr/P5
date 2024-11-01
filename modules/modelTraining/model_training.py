import pandas as pd
import numpy as np
from numpy.typing import NDArray
from time import time

from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.utils import Bunch

from modules.config.config import Config
from modules.config.config_enums import TrainingMethod
from modules.logging import logger
from modules.scoreFunctions.score_function_selector import ScoreFunctionSelector
from modules.types import (
    FittedEstimator,
    UnfittedEstimator,
    CrossValidator,
)


class ModelTrainer:
    _logger = logger
    _config = Config()

    def __init__(
        self,
        estimator: UnfittedEstimator,
        cross_validator: CrossValidator,
        train_x: pd.DataFrame,
        true_y: pd.Series,
    ) -> None:
        """
        Fit an unfitted model and generate a model report of the training session.

        Parameters
        ----------
        estimator : UnfittedEstimator
            The model to train.

        cv : CrossValidator
            The cross-validator to use with the model.

        score_funcs : ModelScoreCallable
            A callable with signature `scorer(estimator, X, y)`.

        train_x : NDArray
            Training feature(s).

        true_y : NDArray
            Target feature, i.e., "Dag".
        """
        self._unfit_estimator = estimator
        self._cross_validator = cross_validator
        self._model_score_funcs = ScoreFunctionSelector.getScoreFuncsModel()
        self._train_x = train_x
        self._true_y = true_y
        self._parent_key = "ModelTraining"
        self._n_jobs = self._config.getValue("n_jobs", "General")
        self._pipeline_report = {
            # "training": {
            "feature_importances": None,  # type: Bunch
            "feature_names_in": None,  # type: NDArray
            "feature_count": None,  # type: int
            # }
        }

    def _checkAllFeaturesPresent(self) -> None:
        """Warn user if they select a training method incompatible with the feature selector"""
        key = "GenericUnivariateSelect"
        parent_key = "DataPreprocessing"
        if self._config.getValue(key, parent_key=parent_key):
            self._logger.warning(
                f"Using a reduced feature set for hyperparameter tuning (this might harm model performance). "
                + f"Please disable '{key}' (within '{parent_key}') in the config when tuning hyperparameters"
            )

    def _compileModelReport(self, estimator: FittedEstimator) -> None:
        """
        Compute and add training results to the model report (a subset of the pipeline report).

        Parameters
        ----------
        estimator : FittedEstimator
            The trained model.
        """
        self._pipeline_report["feature_importances"] = (
            self._permutationFeatureImportance(
                estimator,
                **self._config.getValue(
                    "PermutationFeatureImportance", parent_key=self._parent_key
                ),
            )
        )
        self._pipeline_report["feature_names_in"] = estimator.feature_names_in_
        self._pipeline_report["feature_count"] = estimator.n_features_in_

    def _permutationFeatureImportance(
        self,
        estimator: FittedEstimator,
        **kwargs,
    ) -> Bunch:
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
        estimator : FittedEstimator
            A fitted estimator.

        **kwargs : dict
            Additional parameters defined in the config.

        Returns
        -------
        Bunch
            Dictionary-like object, with the following attributes:
                importances_mean : ndarray, shape (n_features, )
                    Mean of feature importance over `n_repeats`.

                importances_std : ndarray, shape (n_features, )
                    Standard deviation over `n_repeats`.

                importances : ndarray, shape (n_features, n_repeats)
                    Raw permutation importance scores.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
        - https://scikit-learn.org/stable/modules/permutation_importance.html

        """
        # TODO: Implement plots from: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        # TODO: Verify that the model is better than a RNG:
        #       https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html#sphx-glr-auto-examples-model-selection-plot-permutation-tests-for-classification-py
        #       https://scikit-learn.org/stable/modules/cross_validation.html#permutation-test-score
        self._logger.info("Computing permutation feature importances...")
        start_time = time()
        result = permutation_importance(
            estimator,
            self._train_x,
            self._true_y,
            scoring=self._model_score_funcs,
            n_jobs=self._n_jobs,
            **kwargs,
        )
        self._logger.info(
            f"Permutation feature importances completed in {time()-start_time:.3f} s"
        )
        return result

    def _selectEstimatorFromReport(self, training_report: dict) -> FittedEstimator:
        """
        Selects the best estimator from a set of fitted estimators
        by comparing their performance metrics obtained during training.

        Parameters
        ----------
        training_report : dict
            A dict with the following keys:
                estimators : ArrayLike
                    Array of fitted estimators.

                test_scores : dict[str, NDArray]
                    A dict containing test scores for each estimator
                    and performance metric (e.g. score function).

        Returns
        -------
        FittedEstimator
            The best model according to the performance metrics.
        """
        # Create array of zeros with shape equal to the amount of score functions selected
        # Each index "maps" to the equal index in the array of estimators
        test_score_counter = np.zeros(len(training_report["test_scores"].keys()))
        print(test_score_counter)
        test_scores = training_report["test_scores"]  # type: dict[str, NDArray]

        for score_func_name, scores in test_scores.items():
            # Find the estimator that maximises this score function using "argmax"
            # and add a weighted sum to its index mapping
            test_score_counter[scores.argmax()] += 1 * self._config.getValue(
                score_func_name, parent_key="score_function_weights"
            )
        print(test_score_counter)
        # Find the index of the estimator that maxmimises the greatest amount of score functions
        # using argmax and use this index to get the estimator from the estimator array
        estimator = training_report["estimators"][test_score_counter.argmax()]

        # TODO: Print which score functions the selected estimator is maximising
        self._logger.info(f"Selected the best estimator among possible candidates")
        return estimator

    def _fitGridSearchWithCrossValidation(self, **kwargs) -> FittedEstimator:
        """
        GridSearchCV exhaustively generates candidates from a grid of parameter values.

        In other words, it is a brute-force search of the *entire* grid parameter space.
        This ensures that the optimal permutation of hyperparameters are selected to maximise a given score function.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters defined in the config.

        Returns
        -------
        FittedEstimator
            The trained model.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        - https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search
        """

        # TODO: Get model report out of the search

        # SECTION
        # GRID SEARCH (hyperparameter tuning) {On any hyperparamter}

        # Custom refit strategy of a grid search with cross-validation
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

        # Running GridSearchCV using multiple evaluation metrics
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

        # Statistical comparison of models:
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py

        self._checkAllFeaturesPresent()
        gscv = GridSearchCV(
            estimator=self._unfit_estimator,
            param_grid={},  # TODO: Implement
            scoring=self._model_score_funcs,
            n_jobs=self._n_jobs,
            cv=self._cross_validator,
            **kwargs,
        )
        return gscv.best_estimator_

    def _fitRandomSearchWithCrossValidation(self, **kwargs) -> FittedEstimator:
        """
        RandomizedSearchCV implements a randomized search over parameters,
        where each setting is sampled from a distribution over possible parameter values.

        This has two main benefits over an exhaustive search:
        - A budget can be chosen independent of the number of parameters and possible values.
        - Adding parameters that do not influence the performance does not decrease efficiency.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters defined in the config.

        Returns
        -------
        FittedEstimator
            The trained model.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        - https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization
        """
        # TODO: Get model report out of the search
        self._checkAllFeaturesPresent()
        rscv = RandomizedSearchCV(
            estimator=self._unfit_estimator,
            param_distributions={},  # TODO: Implement
            scoring=self._model_score_funcs,
            n_jobs=self._n_jobs,
            cv=self._cross_validator,
            **kwargs,
        )
        return rscv.best_estimator_

    def _fitRFEWithCrossValidation(
        self,
        **kwargs,
    ) -> FittedEstimator:
        """
        A Recursive Feature Elimination (RFE) with automatic
        tuning of the number of features selected with cross-validation (a.k.a. RFECV).

        Parameters
        ----------
        **kwargs : dict
            Additional parameters defined in the config.

        Returns
        -------
        FittedEstimator
            The trained model.

        Raises
        ------
        ValueError
            If muliple score functions are selected (this training method does not support this).

        Links
        -----
        - https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
        """

        # NOTE: Some models do not support this (e.g. GaussianNB)!
        # TODO: Adjust for multiple scoring functions
        self._checkAllFeaturesPresent()
        if len(self._model_score_funcs.keys()) > 1:
            raise ValueError(
                f"Selected model 'RFECV' does not support multiple score functions. Got {self._model_score_funcs.keys()}"
            )

        rfecv = RFECV(
            self._unfit_estimator,
            cv=self._cross_validator,
            scoring=next(iter(self._model_score_funcs.values())),
            n_jobs=self._n_jobs,
            **kwargs,
        ).fit(self._train_x, self._true_y)
        return rfecv.estimator_

    def _fitRFE(self, **kwargs) -> FittedEstimator:
        """
        A Recursive Feature Elimination (RFE) with automatic
        tuning of the number of features selected.
        Parameters
        ----------
        **kwargs : dict
            Additional parameters defined in the config.

        Returns
        -------
        FittedEstimator
            The trained model.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
        """
        # NOTE: Some models do not support this (e.g. GaussianNB)!
        # TODO: Adjust for multiple scoring functions
        self._checkAllFeaturesPresent()
        rfe = RFE(self._unfit_estimator, **kwargs).fit(self._train_x, self._true_y)
        return rfe.estimator_

    def _fitWithCrossValidation(
        self,
    ) -> FittedEstimator:
        """
        Train a set of models using cross-validation.
        A model is fitted for each cross-validation split.

        The best model is selected such that it is maximising the greatest amount of score functions.

        Example
        -------
        1. A model called `DecisionTreeClassifier` is fitted using a cross-validator with 5 splits.
        2. Each cross-validation split produces a fitted estimator of type `DecisionTreeClassifier`.
        3. Each fitted estimator has its own accuracy when measured with some score function (SF).
        4. Inevitably, some estimators will outperform others w.r.t. maximising a given SF.
        5. However, an estimator may perform well on `n` SFs but perform poorly on `m` SFs.
        6. Thus, we want the estimator that outperforms other estimators on as many SFs as possible.
        7. Hence, the best model is selected such that it is maximising the greatest amount of score functions.

        Returns
        -------
        FittedEstimator
            The trained model.

        Links
        -----
        - https://scikit-learn.org/stable/modules/cross_validation.html
        """
        cv_results = cross_validate(
            estimator=self._unfit_estimator,
            X=self._train_x,
            y=self._true_y,
            scoring=self._model_score_funcs,
            cv=self._cross_validator,
            n_jobs=self._n_jobs,
            return_estimator=True,
        )
        print(cv_results)
        training_report = {
            "estimators": cv_results["estimator"],
            "test_scores": {
                score_func_name: cv_results[f"test_{score_func_name}"]
                for score_func_name in self._model_score_funcs.keys()
            },
        }
        return self._selectEstimatorFromReport(training_report)

    def _fit(self) -> FittedEstimator:
        """
        Train a single model directly on the training data.

        Returns
        -------
        FittedEstimator
            The trained model
        """
        return self._unfit_estimator.fit(self._train_x, self._true_y)

    def run(
        self,
    ) -> tuple[FittedEstimator, dict]:
        """
        Train the selected model on the training dataset using
        the settings defined in the config.

        Returns
        -------
        tuple[FittedEstimator, dict]
            [0]: The trained model.
            [1]: The pipeline report with model training results added.

        Raises
        ------
        ValueError
            If the selected training method is invalid.
        """

        training_method = self._config.getValue(
            "training_method", parent_key=self._parent_key
        )

        self._logger.info(f"Training model...")
        start_time = time()

        if training_method == TrainingMethod.FIT.name:
            fitted_estimator = self._fit()
        elif training_method == TrainingMethod.CROSS_VALIDATION.name:
            fitted_estimator = self._fitWithCrossValidation()
        elif training_method == TrainingMethod.RFE.name:
            fitted_estimator = self._fitRFE(
                **self._config.getValue("RFE", self._parent_key)
            )
        elif training_method == TrainingMethod.RFECV.name:
            fitted_estimator = self._fitRFEWithCrossValidation(
                **self._config.getValue("RFECV", self._parent_key),
            )
        elif training_method == TrainingMethod.RANDOM_SEARCH_CV.name:
            random_args = self._config.getValue("RandomizedSearchCV", self._parent_key)
            grid_args = self._config.getValue("GridSearchCV", self._parent_key)
            fitted_estimator = self._fitRandomSearchWithCrossValidation(
                random_args | grid_args
            )
        elif training_method == TrainingMethod.GRID_SEARCH_CV.name:
            fitted_estimator = self._fitGridSearchWithCrossValidation(
                self._config.getValue("GridSearchCV", self._parent_key)
            )
        else:
            raise ValueError(
                f"Invalid training method '{training_method}'. Expected one of {TrainingMethod._member_names_}"
            )

        self._logger.info(f"Model training took {time()-start_time:.3f} s")
        self._compileModelReport(fitted_estimator)
        self._logger.info(
            f"Model training complete! Total training time: {time()-start_time:.3f} s"
        )
        return fitted_estimator, self._pipeline_report
