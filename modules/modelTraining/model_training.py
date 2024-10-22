from time import time
from typing import Any, Callable, TypeAlias, Union
from numpy.typing import NDArray
from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.utils import Bunch

from modules.config.config import Config
from modules.config.config_enums import TrainingMethod
from modules.logging import logger




class ModelTrainer:
    _logger = logger

    def __init__(self) -> None:
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
        self._config = Config()
        self._parent_key = "ModelTraining"

    def _checkAllFeaturesPresent(self) -> None:
        key = "GenericUnivariateSelect"
        parent_key = "DataPreprocessing"
        if self._config.getValue(key, parent_key=parent_key):
            self._logger.warning(
                f"Using a reduced feature set for hyperparameter tuning (this might harm model performance). "
                + f"Please disable '{key}' (within '{parent_key}') in the config when tuning hyperparameters"
            )

    def _compileModelReport(
        self, fitted_estimator: Any, train_x: NDArray, target_y: NDArray
    ) -> None:
        self._model_report["feature_importances"] = self._permutationFeatureImportance(
            fitted_estimator,
            train_x,
            target_y,
            self._config.getValue(
                "PermutationFeatureImportance", parent_key=self._parent_key
            ),
        )
        self._model_report["feature_names_in"] = fitted_estimator.feature_names_in_
        self._model_report["feature_count"] = fitted_estimator.n_features_in_

    def _permutationFeatureImportance(
        self, fitted_estimator: Any, train_x: NDArray, target_y: NDArray, **kwargs
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
        fitted_estimator : Any
            A fitted estimator.

        X : NDArray
            Training data.

        y : NDArray
            Target data.

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
        # TODO: Verify that the model is better than a RNG: https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html#sphx-glr-auto-examples-model-selection-plot-permutation-tests-for-classification-py
        self._logger.info("Computing permutation feature importances...")
        start_time = time()
        result = permutation_importance(
            fitted_estimator,
            train_x,
            target_y,
            n_jobs=-1,
            **kwargs,
        )
        self._logger.info(
            f"Feature importance computation completed in {time()-start_time:.3f} s"
        )
        return result

    def _fitGridSearchCV(
        self, unfit_estimator: Any, train_x: NDArray, target_y: NDArray, **kwargs
    ) -> FitEstimator:
        self._checkAllFeaturesPresent()
        gscv = GridSearchCV()

    def _fitRandomSearchCV(
        self, unfit_estimator: Any, train_x: NDArray, target_y: NDArray, **kwargs
    ) -> FitEstimator:
        self._checkAllFeaturesPresent()
        rscv = RandomizedSearchCV()

    def _fitRFECV(
        self, unfit_estimator: Any, train_x: NDArray, target_y: NDArray, **kwargs
    ) -> FitEstimator:
        self._checkAllFeaturesPresent()
        rfecv = RFECV()

    def _fitRFE(
        self, unfit_estimator: Any, train_x: NDArray, target_y: NDArray, **kwargs
    ) -> FitEstimator:
        self._checkAllFeaturesPresent()
        self._logger.info(f"Fitting model...")
        start_time = time()
        rfe = RFE(unfit_estimator, **kwargs).fit(train_x, target_y)
        self._logger.info(f"Model fitted in {time()-start_time:.3f} s")
        self._compileModelReport(rfe, train_x, target_y)
        return rfe.estimator_

    def _fitCV(
        self,
        unfit_estimator: Any,
        cv: Any,
        train_x: NDArray,
        target_y: NDArray,
        scoring: Callable,
    ) -> FitEstimator:
        cv_results = cross_validate(
            estimator=unfit_estimator,
            X=train_x,
            y=target_y,
            groups=None,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_estimator=True,
        )
        ft = cv_results["estimator"]
        print(ft)  # TODO: Test if we can get fitted estimator like this

    def _fit(
        self,
        unfit_estimator: Any,
        train_x: NDArray,
        target_y: NDArray,
    ) -> FitEstimator:
        self._logger.info(f"Fitting model...")
        start_time = time()
        fitted_estimator = unfit_estimator.fit(train_x, target_y)
        self._logger.info(f"Model fitted in {time()-start_time:.3f} s")
        self._compileModelReport(fitted_estimator, train_x, target_y)
        return fitted_estimator

    def getModelReport(self) -> dict[str, Any]:
        return self._model_report

    # TODO: train_x and target_y MUST include labels (for ALL methods using them)
    def run(
        self,
        unfit_estimator: Any,
        train_x: NDArray,
        target_y: NDArray,
    ) -> Union[]:
        start_time = time()
        training_method = self._config.getValue(
            "training_method", parent_key=self._parent_key
        )

        if training_method == TrainingMethod.FIT:
            fitted_estimator = self._fit(
                unfit_estimator=unfit_estimator, train_x=train_x, target_y=target_y
            )
        elif training_method == TrainingMethod.CROSS_VALIDATION:
            fitted_estimator = self._fitCV(
                unfit_estimator=unfit_estimator,
                train_x=train_x,
                target_y=target_y,
                scoring=getScoring(),
            )
        elif training_method == TrainingMethod.RFE:
            fitted_estimator = self._fitRFE(
                unfit_estimator=unfit_estimator, train_x=train_x, target_y=target_y
            )
        elif training_method == TrainingMethod.RFECV:
            fitted_estimator = self._fitRFECV(
                unfit_estimator=unfit_estimator, train_x=train_x, target_y=target_y
            )
        elif training_method == TrainingMethod.RANDOM_SEARCH_CV:
            fitted_estimator = self._fitRandomSearchCV(
                unfit_estimator=unfit_estimator, train_x=train_x, target_y=target_y
            )
        elif training_method == TrainingMethod.GRID_SEARCH_CV:
            fitted_estimator = self._fitGridSearchCV(
                unfit_estimator=unfit_estimator, train_x=train_x, target_y=target_y
            )
        else:
            raise TypeError(
                f"Invalid training method '{training_method}'. Expected one of {TrainingMethod._member_names_}"
            )

        self._logger.info(
            f"Model training complete! Total training time: {time()-start_time:.3f} s"
        )
        return fitted_estimator
