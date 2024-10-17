from typing import Any

from modules.config.config import Config
from modules.dataPreprocessing.strategy import Strategy


class Processor:
    def __init__(
        self, config_file: Config, pipeline_component: Any, strategy: Strategy
    ):
        self.config_file = config_file
        self.pipeline_component = pipeline_component
        self.strategy = strategy

    def getStrategy(self) -> None:
        return self.strategy

    def setStrategy(self, strategy: Strategy) -> None:
        self.strategy = strategy

    def performAlgorithm(self) -> None:
        self.strategy.performAlgorithm(self.config_file, self.pipeline_component)
