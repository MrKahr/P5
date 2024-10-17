from typing import Any

from modules.config.config import Config
from modules.dataPreprocessing.strategy import Strategy


class Processor:
    def __init__(self, config_file: dict, PipelineComponent: Any, Strategy: Strategy):
        self.config_file = config_file
        self.pipeline_component = PipelineComponent
        self.Strategy = Strategy

    def getStrategy(self) -> None:
        return self.Strategy

    def setStrategy(self, Strategy: Strategy) -> None:
        self.Strategy = Strategy

    def performAlgorithm(self) -> None:
        self.Strategy.performAlgorithm(self.config_file, self.pipeline_component)
