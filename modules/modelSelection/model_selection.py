from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import LogLevel, Model
from modules.gpuBackend.compatibility.config_param_converter import ConfigParamConverter
from modules.gpuBackend.models.mlp_gpu import MLPClassifierGPU
from modules.tools.types import UnfittedEstimator
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
    def _getNeuralNetwork(cls, **kwargs) -> MLPClassifier:
        is_verbose = (
            1
            if PipelineConfig().getValue("loglevel", "General") == LogLevel.DEBUG.name
            else 0
        )

        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False

        if use_cuda:
            return MLPClassifierGPU(
                verbose=is_verbose,
                **ConfigParamConverter.convertToMLPClassifierGPU(kwargs),
            )
        else:
            logger.debug(f"CUDA not available. Switching to CPU")
            return MLPClassifier(**kwargs)

    @classmethod
    def _getNaiveBayes(cls, **kwargs) -> CategoricalNB:
        return CategoricalNB(**kwargs)

    @classmethod
    def getModel(cls) -> UnfittedEstimator:
        """
        Get an unfit instance of the model as specified in the config file.

        Returns
        -------
        UnfittedEstimator
            An unfit instance of the model as specified in the config file.
        """
        cls._config = PipelineConfig()
        parent_key = "ModelSelection"
        selected_model = cls._config.getValue("model", parent_key)

        # Create job count dict to allow combining it with other kwargs when passing args from the config
        # (this is necessary since n_jobs is globally used and not tied to a particular model)
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
                **decision_tree_options | random_forest_options | n_jobs
            )
        elif selected_model == Model.NAIVE_BAYES.name:
            model = cls._getNaiveBayes(
                **cls._config.getValue("CategoricalNaiveBayes", parent_key)
            )
        elif selected_model == Model.NEURAL_NETWORK.name:
            model = cls._getNeuralNetwork(
                **cls._config.getValue("NeuralNetwork", parent_key)
            )
        else:
            raise ValueError(
                f"Invalid model '{selected_model}'. Expected one of {Model._member_names_}"
            )
        logger.info(f"Using model: {type(model).__name__}")
        return model
