from time import time
from typing import Callable
from numpy.typing import NDArray

from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.utils import Bunch

from modules.config.config import Config
from modules.config.config_enums import TrainingMethod
from modules.logging import logger
from modules.types import (
    FittedEstimator,
    UnfittedEstimator,
    CrossValidator,
    ModelScoreCallable,
)


class ModelTrainer:
    _logger = logger

    def __init__(
        self,
        estimator: UnfittedEstimator,
        cv: CrossValidator,
        score_funcs: ModelScoreCallable,  # FIXME: Temporary solution until multi-score is implemented
        train_x: NDArray,
        target_y: NDArray,
    ) -> None:
        """Fit an unfitted model and generate a model report of the training session.

        Parameters
        ----------
        estimator : UnfittedEstimator
            The model to train.

        cv : CrossValidator
            The cross-validator to use with the model.

        score_funcs : ModelScoreCallable
            A callable object / function with signature `scorer(estimator, X, y)`.

        train_x : NDArray
            Training data.

        target_y : NDArray
            Target data.
        """
        self._unfit_estimator = estimator
        self._cv = cv
        self._score_funcs = score_funcs
        self._train_x = train_x
        self._target_y = target_y
        self._config = Config()
        self._parent_key = "ModelTraining"
        self._n_jobs = self._config.getValue("n_jobs", "General")
        self._model_report = {
            "feature_importances": None,  # type: Bunch
            "feature_names_in": None,  # type: NDArray
            "feature_count": None,  # type: int
            "true_positive": None,  # type: float
            "true_negative": None,  # type: float
            "false_positive": None,  # type: float
            "false_negative": None,  # type: float
            "accuracy": None,  # type: float
            "precision": None,  # type: float
            "recall": None,  # type: float
            "specificity": None,  # type: float
        }

    def _checkAllFeaturesPresent(self) -> None:
        key = "GenericUnivariateSelect"
        parent_key = "DataPreprocessing"
        if self._config.getValue(key, parent_key=parent_key):
            self._logger.warning(
                f"Using a reduced feature set for hyperparameter tuning (this might harm model performance). "
                + f"Please disable '{key}' (within '{parent_key}') in the config when tuning hyperparameters"
            )

    def _compileModelReport(self, estimator: FittedEstimator) -> None:
        self._model_report["feature_importances"] = self._permutationFeatureImportance(
            estimator,
            **self._config.getValue(
                "PermutationFeatureImportance", parent_key=self._parent_key
            ),
        )
        self._model_report["feature_names_in"] = estimator.feature_names_in_
        self._model_report["feature_count"] = estimator.n_features_in_

    def _permutationFeatureImportance(
        self, estimator: FittedEstimator, **kwargs
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
            self._target_y,
            n_jobs=self._n_jobs,
            **kwargs,
        )
        self._logger.info(
            f"Feature importance computation completed in {time()-start_time:.3f} s"
        )
        return result

    def _fitGridSearchCV(self, **kwargs) -> FittedEstimator:
        # TODO: Get model report out of the search
        self._checkAllFeaturesPresent()
        gscv = GridSearchCV(
            estimator=self._unfit_estimator,
            param_grid={},  # TODO: Implement
            scoring=self._score_funcs,
            n_jobs=self._n_jobs,
            cv=self._cv,
            **kwargs,
        )
        return gscv.best_estimator_

    def _fitRandomSearchCV(self, **kwargs) -> FittedEstimator:
        # TODO: Get model report out of the search
        self._checkAllFeaturesPresent()
        rscv = RandomizedSearchCV(
            estimator=self._unfit_estimator,
            param_distributions={},  # TODO: Implement
            scoring=self._score_funcs,
            n_jobs=self._n_jobs,
            cv=self._cv,
            **kwargs,
        )
        return rscv.best_estimator_

    def _fitRFECV(
        self,
        **kwargs,
    ) -> FittedEstimator:
        # NOTE: Some models do not support this (e.g. GaussianNB)!
        self._checkAllFeaturesPresent()
        rfecv = RFECV(
            self._unfit_estimator,
            cv=self._cv,
            scoring=self._score_funcs,
            n_jobs=self._n_jobs,
            **kwargs,
        )
        return rfecv.estimator_

    def _fitRFE(self, **kwargs) -> FittedEstimator:
        # NOTE: Some models do not support this (e.g. GaussianNB)!
        self._checkAllFeaturesPresent()
        rfe = RFE(self._unfit_estimator, **kwargs).fit(self._train_x, self._target_y)
        return rfe.estimator_

    def _fitCV(
        self,
        cv: CrossValidator,
        scoring: Callable,
    ) -> FittedEstimator:
        cv_results = cross_validate(
            estimator=self._unfit_estimator,
            X=self._train_x,
            y=self._target_y,
            scoring=scoring,
            cv=cv,
            n_jobs=self._n_jobs,
            return_estimator=True,
        )
        ft = cv_results["estimator"]
        print(ft)  # TODO: Test if we can get fitted estimator like this

    def _fit(self) -> FittedEstimator:
        return self._unfit_estimator.fit(self._train_x, self._target_y)

    def run(
        self,
    ) -> tuple[FittedEstimator, dict]:
        training_method = self._config.getValue(
            "training_method", parent_key=self._parent_key
        )
        self._logger.info(f"Training model...")
        start_time = time()

        if training_method == TrainingMethod.FIT.name:
            fitted_estimator = self._fit()
        elif training_method == TrainingMethod.CROSS_VALIDATION:
            fitted_estimator = self._fitCV(
                cv=self._cv,
                scoring=self._score_funcs,
            )
        elif training_method == TrainingMethod.RFE.name:
            fitted_estimator = self._fitRFE(
                **self._config.getValue("RFE", self._parent_key)
            )
        elif training_method == TrainingMethod.RFECV.name:
            fitted_estimator = self._fitRFECV(
                cv=self._cv,
                scoring=self._score_funcs,
                **self._config.getValue("RFECV", self._parent_key),
            )
        elif training_method == TrainingMethod.RANDOM_SEARCH_CV.name:
            random_args = self._config.getValue("RandomizedSearchCV", self._parent_key)
            grid_args = self._config.getValue("GridSearchCV", self._parent_key)
            fitted_estimator = self._fitRandomSearchCV(random_args | grid_args)
        elif training_method == TrainingMethod.GRID_SEARCH_CV.name:
            fitted_estimator = self._fitGridSearchCV(
                self._config.getValue("GridSearchCV", self._parent_key)
            )
        else:
            raise TypeError(
                f"Invalid training method '{training_method}'. Expected one of {TrainingMethod._member_names_}"
            )

        self._logger.info(f"Model training in {time()-start_time:.3f} s")
        self._compileModelReport(fitted_estimator)
        self._logger.info(
            f"Model training complete! Total training time: {time()-start_time:.3f} s"
        )
        return fitted_estimator, self._model_report
