from typing import Union
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from modules.config.config import Config
from modules.config.config_enums import Model
from modules.logging import logger


class ModelSelector:
    """
    Model selection for the pipeline.
    Creates an returns an instance of the model specified in the config file.
    """

    @classmethod
    def _getDecisionTree(
        cls,
        kwargs,
    ) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**kwargs)

    @classmethod
    def _getRandomForest(cls, kwargs) -> RandomForestClassifier:
        return RandomForestClassifier(**kwargs)

    @classmethod
    def _getNeuralNetwork(cls, kwargs) -> None:
        raise NotImplementedError()

    @classmethod
    def _getSupportVectorMachine(cls, kwargs) -> None:
        raise NotImplementedError()
        return SVC(**kwargs)

    @classmethod
    def _getNaiveBayes(cls, kwargs) -> None:
        raise NotImplementedError()
        return GaussianNB(**kwargs)

    @classmethod
    def run(cls) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
        """Get an unfit instance of the model as specified in the config file.

        Returns
        -------
        Union[DecisionTreeClassifier, RandomForestClassifier]
            An unfit instance of the model as specified in the config file.
        """
        cls._config = Config()
        parent_key = "ModelSelection"
        selected_model = cls._config.getValue("model", parent_key)

        if selected_model == Model.DECISION_TREE.name:
            model = cls._getDecisionTree(
                cls._config.getValue("DecisionTree", parent_key)
            )
        elif selected_model == Model.RANDOM_FOREST.name:
            decision_tree_options = cls._config.getValue("DecisionTree", parent_key)
            random_forest_options = cls._config.getValue("RandomForest", parent_key)
            model = cls._getRandomForest(decision_tree_options | random_forest_options)
        elif selected_model == Model.NAIVE_BAYES.name:
            model = cls._getNaiveBayes()
        elif selected_model == Model.SUPPORT_VECTOR.name:
            model = cls._getSupportVectorMachine()
        elif selected_model == Model.NEURAL_NETWORK.name:
            model = cls._getNeuralNetwork()
        else:
            raise NotImplementedError(f"No support for model '{selected_model}'")
        logger.info(f"ModelSelector is done")
        return model
