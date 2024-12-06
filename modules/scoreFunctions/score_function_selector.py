from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    make_scorer,
)

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import FeatureScoreFunc, ModelScoreFunc
from modules.logging import logger
from modules.scoreFunctions.score_functions import (
    FeatureSelectScoreFunctions,
    ModelScoreFunctions,
)
from modules.tools.types import (
    ModelScoreCallable,
    FeatureSelectScoreCallable,
)


# FIXME: Optimize code structure of class
class ScoreFunctionSelector:
    """
    The class responsible for collecting the desired score functions
    for model training and feature selectiom, as defined in the config,
    and standardise their required parameters to allow their use freely
    across the entire pipeline.
    """

    _config = PipelineConfig()

    # We use a cache to avoid repeated computation when picking score functions according to the config
    _cache = {
        "model": None,
        "feature_select": None,
    }

    @classmethod
    def getScoreFuncsModel(
        cls,
    ) -> dict[str, ModelScoreCallable]:
        """
        Create a dict of the model score functions where each key is the
        lowercase name of the selected score functions, as defined in the config.

        Returns
        -------
        dict[str, ModelScoreCallable]
            A dict with each model score function name as keys
            and the model score callable as values.

        Raises
        ------
        ValueError
            If the selected model score function is invalid.
        """
        if cls._cache["model"] is None:
            parent_key = "ModelTraining"
            score_funcs = cls._config.getValue("score_functions", parent_key)

            # The score function in the config can be either a list or single value; we ensure it's always a list
            if not isinstance(score_funcs, list):
                score_funcs = [score_funcs]

            selected_score_funcs = {}
            for score_func in score_funcs:

                # Add threshold score function
                if score_func in [
                    ModelScoreFunc.THRESHOLD.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.THRESHOLD.name.lower(): make_scorer(
                            ModelScoreFunctions.threshold,
                            threshold=cls._config.getValue(
                                "threshold", parent_key="score_function_params"
                            ),
                        )
                    }

                # Add distance score function
                if score_func in [
                    ModelScoreFunc.DISTANCE.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.DISTANCE.name.lower(): make_scorer(
                            ModelScoreFunctions.distance
                        )
                    }

                # Add accuracy score function
                if score_func in [
                    ModelScoreFunc.EXACT_ACCURACY.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.EXACT_ACCURACY.name.lower(): make_scorer(
                            accuracy_score
                        )
                    }

                # Add balanced accuracy score function
                if score_func in [
                    ModelScoreFunc.BALANCED_ACCURACY.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.BALANCED_ACCURACY.name.lower(): make_scorer(
                            balanced_accuracy_score, adjusted=True
                        )
                    }

                # We didn't find any score functions to use. Thus, the config's value must be invalid
                if not selected_score_funcs:
                    raise ValueError(
                        f"Invalid model score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                    )

            logger.info(
                f"Using model score functions: [{", ".join(selected_score_funcs.keys())}]"
            )
            cls._cache["model"] = selected_score_funcs

        return cls._cache["model"]

    @classmethod
    def getScoreFuncFeatureSelect(
        cls,
    ) -> FeatureSelectScoreCallable:
        """
        Create the feature selection function as specified in the config.

        Returns
        -------
        FeatureSelectScoreCallable
            The feature selection function as a callable.

        Raises
        ------
        ValueError
            If the selected feature selection function is invalid.
        """
        if cls._cache["feature_select"] is None:
            parent_key = "FeatureSelection"
            score_func = cls._config.getValue("score_function", parent_key)

            # Chi-squared function
            if score_func == FeatureScoreFunc.CHI2.name:
                selected_score_func = FeatureSelectScoreFunctions.chi2Independence

            # ANOVA-f function
            elif score_func == FeatureScoreFunc.ANOVA_F.name:
                selected_score_func = FeatureSelectScoreFunctions.fClassifIndependence

            # Mutual information function
            elif score_func == FeatureScoreFunc.MUTUAL_INFO_CLASSIFER.name:
                # Lambda function needed to supply kwargs to the function when called elsewhere
                # (as the users of this callable cannot supply such args)
                selected_score_func = lambda train_x, true_y: FeatureSelectScoreFunctions.mutualInfoClassif(
                    train_x=train_x,
                    true_y=true_y,
                    **cls._config.getValue("MutualInfoClassifArgs", parent_key),
                )
            else:
                raise ValueError(
                    f"Invalid feature select score function '{score_func}'. Expected one of {FeatureScoreFunc._member_names_}"
                )
            logger.info(f"Using feature select score function: '{score_func.lower()}'")

            cls._cache["feature_select"] = selected_score_func
        return cls._cache["feature_select"]

    @classmethod
    def getPriorityScoreFunc(self) -> ModelScoreCallable:
        """
        Get the score function with the heighest weight assigned to it, as defined in the config.
        In case of equal weights for multiple score functions, the last defined is selected.

        Returns
        -------
        ModelScoreCallable
            The score function with the heighest weight.
        """
        score_func_weights = self._config.getValue(
            "score_function_weights", "ModelTraining"
        )

        largest_weight = (None, 0)
        for k, v in score_func_weights.items():
            if k in self.getScoreFuncsModel().keys() and v >= largest_weight[1]:
                largest_weight = (k, v)
        return self.getScoreFuncsModel()[largest_weight[0]]
