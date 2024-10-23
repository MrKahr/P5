from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from modules.config.config import Config
from modules.config.config_enums import Model
from modules.types import UnfittedEstimator


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
    def getModel(
        cls,
    ) -> UnfittedEstimator:
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
            return cls._getDecisionTree(
                **cls._config.getValue("DecisionTree", parent_key)
            )
        elif selected_model == Model.RANDOM_FOREST.name:
            dt = cls._config.getValue("DecisionTree", parent_key)  # type: dict
            rf = cls._config.getValue("RandomForest", parent_key)  # type: dict
            return cls._getRandomForest(**{dt | rf | n_jobs})
        elif selected_model == Model.NAIVE_BAYES.name:
            return cls._getNaiveBayes(
                **cls._config.getValue("GaussianNaiveBayes", parent_key)
            )
        elif selected_model == Model.NEURAL_NETWORK.name:
            return cls._getNeuralNetwork()
        else:
            raise TypeError(
                f"Invalid model '{selected_model}'. Expected one of {Model._member_names_}"
            )

    @classmethod
    def run(self) -> None:
        print(f"{__name__}is run")
