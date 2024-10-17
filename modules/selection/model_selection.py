from typing import Union
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from modules.config.config import Config
from modules.config.config_enums import Model


class ModelSelector:

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
    def getModel(
        cls,
    ) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
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
            return cls._getDecisionTree(
                cls._config.getValue("DecisionTree", parent_key)
            )
        elif selected_model == Model.RANDOM_FOREST.name:
            dt = cls._config.getValue("DecisionTree", parent_key)
            rf = cls._config.getValue("RandomForest", parent_key)
            return cls._getRandomForest(dt | rf)
        elif selected_model == Model.NAIVE_BAYES.name:
            return cls._getNaiveBayes()
        elif selected_model == Model.SUPPORT_VECTOR.name:
            return cls._getSupportVectorMachine()
        elif selected_model == Model.NEURAL_NETWORK.name:
            return cls._getNeuralNetwork()
        else:
            raise NotImplementedError(f"No support for model '{selected_model}'")
