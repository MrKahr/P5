from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from modules.config.config import Config
from modules.config.config_enums import CrossValidator, TrainingMethod
from modules.logging import logger


class CrossValidationSelector:
    """
    Cross-validation selection for the pipeline.
    Creates an returns an instance of the cross-validator specified in the config file.
    """

    @classmethod
    def _getStratifiedKFold(cls, **kwargs) -> StratifiedKFold:
        return StratifiedKFold(**kwargs)

    @classmethod
    def _getTimeSeriesSplit(cls, **kwargs) -> TimeSeriesSplit:
        return TimeSeriesSplit(**kwargs)

    # TODO: Implement: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold

    @classmethod
    def getCrossValidator(
        cls,
    ) -> CrossValidator | None:
        """
        Get an instance of the cross-validator as specified in the config file.

        Returns
        -------
        CrossValidator | None
            An instance of the cross-validator as specified in the config file.

        Raises
        ------
        ValueError
            If the selected cross-validator is invalid.
        """
        cls._config = Config()
        parent_key = "CrossValidationSelection"
        selected_cross_validator = cls._config.getValue("cross_validator", parent_key)

        # Some model training methods are NOT compatible with cross-validation.
        # Thus, we will disable cross-validation if an incompatible training method is selected in the config.
        cross_validator_not_applicable = cls._config.getValue(
            "training_method", "ModelTraining"
        ) in [TrainingMethod.FIT.name]

        # Find a cross-validator to use according to the config
        if selected_cross_validator == None or cross_validator_not_applicable:
            cross_validator = None
        elif selected_cross_validator == CrossValidator.STRATIFIED_KFOLD.name:
            cross_validator = cls._getStratifiedKFold(
                **cls._config.getValue("StratifiedKFold", parent_key)
            )
        elif selected_cross_validator == CrossValidator.TIMESERIES_SPLIT.name:
            cross_validator = cls._getTimeSeriesSplit(
                **cls._config.getValue("TimeSeriesSplit", parent_key)
            )
        else:
            raise ValueError(
                f"Invalid cross-validator '{selected_cross_validator}'. Expected one of {CrossValidator._member_names_}"
            )

        if cross_validator:
            logger.info(f"Using cross-validator: {type(cross_validator).__name__}")
        else:
            logger.info(f"Skipping cross-validation")
        return cross_validator
