from typing import Any
from modules.config.config import Config
from modules.dataPreprocessing.strategy import Strategy
from modules.logging import logger


class StrategySkip(Strategy):

    def performAlgorithm(self, config_file: dict, PipelineComponent: Any) -> None:
        logger.warning(f"Skipping step {config_file}")
