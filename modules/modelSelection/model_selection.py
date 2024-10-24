from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from modules.config.config import Config
from modules.config.config_enums import Model
from modules.types import UnfittedEstimator
from modules.logging import logger


class ModelSelector:
    """
    Model selection for the pipeline.
    Creates an returns an instance of the model specified in the config file.
    """

    @classmethod
    def _getDecisionTree(
        cls,
        **kwargs,
    ) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**kwargs)

    @classmethod
    def _getRandomForest(cls, **kwargs) -> RandomForestClassifier:
        return RandomForestClassifier(**kwargs)

    @classmethod
    def _getNeuralNetwork(cls, **kwargs) -> None:
        raise NotImplementedError()

    @classmethod
    def _getNaiveBayes(cls, **kwargs) -> GaussianNB:
        return GaussianNB(**kwargs)

    @classmethod
    def run(cls) -> UnfittedEstimator:
        """Get an unfit instance of the model as specified in the config file.

        Returns
        -------
        UnfittedEstimator
            An unfit instance of the model as specified in the config file.
        """
        cls._config = Config()
        parent_key = "ModelSelection"
        selected_model = cls._config.getValue("model", parent_key)
        n_jobs = {"n_jobs": cls._config.getValue("n_jobs", "General")}

        if selected_model == Model.DECISION_TREE.name:
            model = cls._getDecisionTree(
                **cls._config.getValue("DecisionTree", parent_key)
            )
        elif selected_model == Model.RANDOM_FOREST.name:
            decision_tree_options = cls._config.getValue(
                "DecisionTree", parent_key
            )  # type: dict
            random_forest_options = cls._config.getValue(
                "RandomForest", parent_key
            )  # type: dict
            model = cls._getRandomForest(
                **{decision_tree_options | random_forest_options | n_jobs}
            )
        elif selected_model == Model.NAIVE_BAYES.name:
            model = cls._getNaiveBayes(
                **cls._config.getValue("GaussianNaiveBayes", parent_key)
            )
        elif selected_model == Model.NEURAL_NETWORK.name:
            model = cls._getNeuralNetwork()
        else:
            raise TypeError(
                f"Invalid model '{selected_model}'. Expected one of {Model._member_names_}"
            )
        logger.info(f"ModelSelector is done")
        return model
