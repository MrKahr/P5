from typing import Any
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.strategy import Strategy


class StrategyparseConfig(Strategy):
    def performAlgorithm(self, config_file: dict, PiplineComponent: Any) -> None:
        for key, value in config_file.items():
            currentCallable = getattr(PiplineComponent, key)
            print(value)
            if value:
                currentCallable(value)
            else:
                currentCallable()
