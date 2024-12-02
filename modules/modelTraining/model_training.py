from typing import Callable, Union
import pandas as pd
import numpy as np
from numpy.typing import NDArray, ArrayLike

from time import time

from sklearn import model_selection
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.utils import Bunch

from modules.config.config import Config
from modules.config.utils.config_enums import TrainingMethod, Model
from modules.logging import logger
from modules.modelTraining.param_grids import ParamGridGenerator
from modules.scoreFunctions.score_function_selector import ScoreFunctionSelector
from modules.tools.random import RNG
from modules.tools.types import (
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
        unsplit_x: pd.DataFrame,
        unsplit_y: pd.Series,
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

        unsplit_x : NDArray
            Training feature(s).

        unsplit_y : NDArray
            Target feature, i.e., "Dag".
        """
        self._unfit_estimator = estimator
        self._cross_validator = cross_validator
        self._model_score_funcs = ScoreFunctionSelector.getScoreFuncsModel()
        self._priority_score_func = (
            ScoreFunctionSelector.getPriorityScoreFunc()
        )  # Temporary
        self._parent_key = "ModelTraining"
        self._n_jobs = self._config.getValue("n_jobs", "General")
        self._training_method = None  # Created during training
        self._pipeline_report = None  # Created during training

        # *_x == pd.DataFrame, *_y == pd.Series
        self._train_x, self._test_x, self._train_true_y, self._test_true_y = (
            model_selection.train_test_split(
                unsplit_x,
                unsplit_y,
                train_size=0.90,
                random_state=111,
                shuffle=True,
                stratify=unsplit_y,
            )
        )

    def _reduceFeatures(self, selected_feature_names: list[str]) -> None:
        """
        Removes features determined by feature selection carried out during model training.
        This is applied to the instance variables `train_x` and `test_x`.

        Parameters
        ----------
        selected_feature_names : list[str]
            Features to keep.
        """
        self._train_x = self._train_x[selected_feature_names]
        self._test_x = self._test_x[selected_feature_names]

    def _checkAllFeaturesPresent(self) -> None:
        """Warn user if they select a training method incompatible with the feature selector"""
        key = "GenericUnivariateSelect"
        parent_key = "DataPreprocessing"
        if self._config.getValue(
            "UseFeatureSelector", parent_key="General"
        ) and self._config.getValue(key, parent_key=parent_key):
            self._logger.warning(
                f"Using a reduced feature set for hyperparameter tuning (this might harm model performance). "
                + f"Please disable feature selection when tuning hyperparameters"
            )

    def _checkModelCompatibility(self) -> FittedEstimator | None:
        model_type = self._config.getValue("model", "ModelSelection")

        if model_type == Model.NAIVE_BAYES.name and self._training_method in [
            TrainingMethod.RFE.name,
            TrainingMethod.RFECV.name,
        ]:
            self._logger.warning(
                f"Model {type(self._unfit_estimator).__name__} does not support RFE-based training. Switching to SFS"
            )
            return self._fitSFS(self._train_x, self._train_true_y)

    def _compileModelReport(self, estimator: FittedEstimator) -> None:
        """
        Compute and add training results to the pipeline report.

        Parameters
        ----------
        estimator : FittedEstimator
            The trained model.
        """
        self._pipeline_report = {
            "estimator": estimator,
            "feature_importances": (
                self._permutationFeatureImportance(
                    estimator,
                    **self._config.getValue(
                        "PermutationFeatureImportance", parent_key=self._parent_key
                    ),
                )
            ),
            "feature_names_in": self._train_x.columns.values,
            "feature_count": len(self._train_x.columns.values),
            "train_x": self._train_x,  # type: pd.DataFrame
            "train_true_y": self._train_true_y,  # type: pd.Series
            "test_x": self._test_x,  # type: pd.DataFrame
            "test_true_y": self._test_true_y,  # type: pd.Series
        }

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
            self._train_true_y,
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
                    for each score function in use.

                    This dict has the following keys:
                        <score_func_name>

        Returns
        -------
        FittedEstimator
            The best model according to the performance metrics.
        """
        # Create array of zeros with shape equal to the amount of estimators fitted
        # Each index "maps" to the equal index in the array of estimators
        estimator_count = len(training_report["estimators"])
        test_score_counter = np.zeros(estimator_count)
        test_scores = training_report["test_scores"]  # type: dict[str, NDArray]

        for score_func_name, scores in test_scores.items():
            # Find the estimator that maximises this score function using "argmax"
            # and add a weighted sum to its index mapping
            test_score_counter[scores.argmax()] += 1 * self._config.getValue(
                score_func_name, parent_key="score_function_weights"
            )
        # Find the index of the estimator that maxmimises the greatest amount of score functions
        # using argmax and use this index to get the estimator from the estimator array
        best_index = test_score_counter.argmax()
        estimator = training_report["estimators"][best_index]

        # We add one to index when logging to match estimator_count (as 1-indexed)
        self._logger.debug(
            f"Selected estimator {best_index + 1} among {estimator_count} possible candidates"
        )
        return estimator

    def _fitGridSearchWithCrossValidation(
        self,
        x: Union[pd.DataFrame, ArrayLike],
        y: Union[pd.Series, ArrayLike],
        refit: Union[bool, str, Callable],
        **kwargs,
    ) -> FittedEstimator:
        """
        GridSearchCV exhaustively generates candidates from a grid of parameter values.

        In other words, it is a brute-force search of the *entire* grid parameter space.
        This ensures that the optimal permutation of hyperparameters are selected to maximise a given score function.

        Parameters
        ----------
        refit : bool | str | Callable
            Refit an estimator using the best found parameters on the whole dataset.
            Only present here to create lowercase string.

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

        # Custom refit strategy of a grid search with cross-validation
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

        # Running GridSearchCV using multiple evaluation metrics
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py

        # Statistical comparison of models:
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py

        self._checkAllFeaturesPresent()

        gscv = GridSearchCV(
            estimator=self._unfit_estimator,
            param_grid=ParamGridGenerator(len(self._train_x.columns)).getParamGrid(),
            scoring=self._model_score_funcs,
            n_jobs=self._n_jobs,
            refit=refit.lower() if isinstance(refit, str) else refit,
            cv=self._cross_validator,
            **kwargs,
        ).fit(x, y)
        # Best estimator attribute is the best model found by gridsearch
        # Alternative predict method uses this attribute but obscures estimator usage

        # TODO: Save relevant info
        # print(gscv.cv_results_)

        return gscv.best_estimator_

    def _fitRandomSearchWithCrossValidation(
        self,
        x: Union[pd.DataFrame, ArrayLike],
        y: Union[pd.Series, ArrayLike],
        random_state: int,
        **kwargs,
    ) -> FittedEstimator:
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
            param_distributions=ParamGridGenerator(
                len(self._train_x.columns)
            ).getRandomParamGrid(),
            scoring=self._model_score_funcs,
            n_jobs=self._n_jobs,
            cv=self._cross_validator,
            random_state=RNG(random_state),
            **kwargs,
        ).fit(x, y)
        return rscv.best_estimator_

    def _fitSFS(
        self, x: Union[pd.DataFrame, ArrayLike], y: Union[pd.Series, ArrayLike]
    ) -> FittedEstimator:
        """
        The Sequential Feature Selector adds (forward selection) or
        removes (backward selection) features to form a feature subset in a greedy fashion.
        At each stage, this estimator chooses the best feature to add or remove based on the
        cross-validation score of an estimator.

        NOTE
        ----
        - Do not call this directly in run()!
        - This method is meant as a substitute for models incompatible with RFE-based training.

        Returns
        -------
        FittedEstimator
            The fitted estimator.

        Links
        -----
        - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
        """
        sfs = SequentialFeatureSelector(
            self._unfit_estimator,
            n_features_to_select="auto",  # Feature selection depends on `tol`
            tol=0.02,  # Tolerance for score improvement that removing a feature must satisfy
            direction="backward",  # Start with all features
            scoring=self._priority_score_func,
            cv=self._cross_validator,
            n_jobs=self._n_jobs,
        ).fit(x, y)
        self._reduceFeatures(sfs.get_feature_names_out())
        return self._fitWithCrossValidation(self._train_x, self._train_true_y)

    def _fitRFEWithCrossValidation(
        self,
        x: Union[pd.DataFrame, ArrayLike],
        y: Union[pd.Series, ArrayLike],
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
        # If the estimator was fitted with another training method
        # because the estimator was incompatible with RFE-based training,
        # then return that estimator.
        estimator = self._checkModelCompatibility()
        if estimator is not None:
            return estimator

        self._checkAllFeaturesPresent()
        # Train an estimator with RFECV for each score function,
        # then select the estimator that maximises the highest weighted score function.
        # An additional step is taken to preserve feature names, as that is missing with the default
        # RFECV implementation.
        training_report = {"estimators": [], "test_scores": {}, "feature_names_out": []}
        for func_name, score_func in self._model_score_funcs.items():
            self._logger.info(f"Running RFECV with score function '{func_name}'")
            rfecv = RFECV(
                self._unfit_estimator,
                cv=self._cross_validator,
                scoring=score_func,
                n_jobs=self._n_jobs,
                **kwargs,
            ).fit(x, y)

            # Create the model report for each score function
            mean_test_score = rfecv.cv_results_["mean_test_score"]
            training_report["estimators"].append(rfecv.estimator_)
            training_report["test_scores"] |= {
                func_name: mean_test_score[np.argmax(mean_test_score)]
            }
            training_report["feature_names_out"].append(rfecv.get_feature_names_out())

        # Get index of the best estimator from the training report
        index = training_report["estimators"].index(
            self._selectEstimatorFromReport(training_report)
        )
        # Use index to get the selected features from the best model
        selected_features = training_report["feature_names_out"][index]
        self._reduceFeatures(selected_features)

        # Fit new model with the selected features AND feature names
        self._logger.info("Training model with optimal feature count")
        return self._fitWithCrossValidation(self._train_x, y)

    def _fitRFE(
        self,
        x: Union[pd.DataFrame, ArrayLike],
        y: Union[pd.Series, ArrayLike],
        **kwargs,
    ) -> FittedEstimator:
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
        # If the estimator was fitted with another training method
        # because the estimator was incompatible with RFE-based training,
        # then return that estimator.
        estimator = self._checkModelCompatibility()
        if estimator is not None:
            return estimator

        self._checkAllFeaturesPresent()
        rfe = RFE(self._unfit_estimator, **kwargs).fit(x, y)
        self._reduceFeatures(rfe.get_feature_names_out())
        return self._fitWithCrossValidation(self._train_x, self._train_true_y)

    def _fitWithCrossValidation(
        self, x: Union[pd.DataFrame, ArrayLike], y: Union[pd.Series, ArrayLike]
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
            X=x,
            y=y,
            scoring=self._model_score_funcs,
            cv=self._cross_validator,
            n_jobs=self._n_jobs,
            return_estimator=True,
        )
        self._logger.debug(
            f"Cross-Validation results:\n\t{"\n\t".join([f"{k}: {v.__repr__()}" for k, v in cv_results.items()])}"
        )
        training_report = {
            "estimators": cv_results["estimator"],
            "test_scores": {
                score_func_name: cv_results[f"test_{score_func_name}"]
                for score_func_name in self._model_score_funcs.keys()
            },
        }
        return self._selectEstimatorFromReport(training_report)

    def _fit(
        self, x: Union[pd.DataFrame, ArrayLike], y: Union[pd.Series, ArrayLike]
    ) -> FittedEstimator:
        """
        Train a single model directly on the training data.

        Parameters
        ----------
        x : pd.DataFrame | ArrayLike
            Training data with features.

        y : pd.Series | ArrayLike
            Target label.

        Returns
        -------
        FittedEstimator
            The trained model
        """
        return self._unfit_estimator.fit(x, y)

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

        self._training_method = self._config.getValue(
            "training_method", parent_key=self._parent_key
        )

        self._logger.info(f"Training model...")
        start_time = time()

        if self._training_method == TrainingMethod.FIT.name:
            fitted_estimator = self._fit(self._train_x, self._train_true_y)
        elif self._training_method == TrainingMethod.CROSS_VALIDATION.name:
            fitted_estimator = self._fitWithCrossValidation(
                self._train_x, self._train_true_y
            )
        elif self._training_method == TrainingMethod.RFE.name:
            fitted_estimator = self._fitRFE(
                self._train_x,
                self._train_true_y,
                **self._config.getValue("RFE", self._parent_key),
            )
        elif self._training_method == TrainingMethod.RFECV.name:
            fitted_estimator = self._fitRFEWithCrossValidation(
                self._train_x,
                self._train_true_y,
                **self._config.getValue("RFECV", self._parent_key),
            )
        elif self._training_method == TrainingMethod.RANDOM_SEARCH_CV.name:
            random_args = self._config.getValue("RandomizedSearchCV", self._parent_key)
            grid_args = self._config.getValue("GridSearchCV", self._parent_key)
            fitted_estimator = self._fitRandomSearchWithCrossValidation(
                self._train_x, self._train_true_y, **random_args | grid_args
            )
        elif self._training_method == TrainingMethod.GRID_SEARCH_CV.name:
            fitted_estimator = self._fitGridSearchWithCrossValidation(
                self._train_x,
                self._train_true_y,
                **self._config.getValue("GridSearchCV", self._parent_key),
            )
        else:
            raise ValueError(
                f"Invalid training method '{self._training_method}'. Expected one of {TrainingMethod._member_names_}"
            )

        self._logger.info(f"Model training took {time()-start_time:.3f} s")
        self._compileModelReport(fitted_estimator)
        self._logger.info(
            f"Model training complete! Total running time: {time()-start_time:.3f} s"
        )
        return self._pipeline_report
