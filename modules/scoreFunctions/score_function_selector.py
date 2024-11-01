from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    make_scorer,
)

from modules.config.config import Config
from modules.config.config_enums import FeatureScoreFunc, ModelScoreFunc
from modules.logging import logger
from modules.scoreFunctions.score_functions import (
    FeatureSelectScoreFunctions,
    ModelScoreFunctions,
)
from modules.types import (
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

    _config = Config()

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
                    ModelScoreFunc.ACCURACY.name,
                    ModelScoreFunc.ALL.name,
                ]:
                    selected_score_funcs |= {
                        ModelScoreFunc.ACCURACY.name.lower(): make_scorer(
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
                            balanced_accuracy_score
                        )
                    }

                # Add explained variance score function
                # REVIEW: Do we want regression score functions?
                # if score_func in [
                #     ModelScoreFunc.EXPLAINED_VARIANCE.name,
                #     ModelScoreFunc.ALL.name,
                # ]:
                #     selected_score_funcs |= {
                #         ModelScoreFunc.EXPLAINED_VARIANCE.name.lower(): make_scorer(
                #             explained_variance_score
                #         )
                #     }

                # We didn't find any score functions to use. Thus, the config's value must be invalid
                if not selected_score_funcs:
                    raise ValueError(
                        f"Invalid model score function '{score_funcs}'. Expected one of {ModelScoreFunc._member_names_}"
                    )

            logger.info(
                f"Using model score functions: '{", ".join(selected_score_funcs.keys())}'"
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
            score_func = cls._config.getValue("score_functions", parent_key)

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
