from typing import Union
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from modules.config.config import Config
from modules.config.config_enums import CrossValidator


class CrossValidationSelector:
    """
    Cross-validation selection for the pipeline.
    Creates an returns an instance of the cross-validator specified in the config file.
    """

    @classmethod
    def _getStratifiedKFold(cls, kwargs) -> StratifiedKFold:
        return StratifiedKFold(**kwargs)

    @classmethod
    def _getTimeSeriesSplit(cls, kwargs) -> TimeSeriesSplit:
        return TimeSeriesSplit(**kwargs)

    @classmethod
    # NOTE: Union class is a union of all types in "[]"
    def getCrossValidator(
        cls,
    ) -> Union[StratifiedKFold, TimeSeriesSplit, None]:
        """Get an instance of the cross-validator as specified in the config file.

        Returns
        -------
        Union[StratifiedKFold, TimeSeriesSplit, None]
            An instance of the cross-validator as specified in the config file.
        """
        cls._config = Config()
        parent_key = "CrossValidationSelection"
        selected_cv = cls._config.getValue("cross_validator", parent_key)

        if selected_cv == None:
            return
        elif selected_cv == CrossValidator.STRATIFIED_KFOLD.name:
            return cls._getStratifiedKFold(
                cls._config.getValue("StratifiedKFold", parent_key)
            )
        elif selected_cv == CrossValidator.TIMESERIES_SPLIT.name:
            return cls._getTimeSeriesSplit(
                cls._config.getValue("TimeSeriesSplit", parent_key)
            )
        else:
            raise NotImplementedError(f"No support for cross-validator '{selected_cv}'")
