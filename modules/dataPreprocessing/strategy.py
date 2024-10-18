from pyparsing import Any, abstractmethod

from modules.config.config import Config


class Strategy:
    @abstractmethod
    def performAlgorithm(self, config_file: dict, PipelineComponent: Any) -> None:
        pass

    @abstractmethod
    def fitModel(self, estimator, features, target) -> Any:
        pass
