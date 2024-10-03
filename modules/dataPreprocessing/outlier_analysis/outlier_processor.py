from typing import Callable
from modules.dataPreprocessing.preprocessor import DataPreprocessor


class OutlierProcessor(DataPreprocessor):
    def __init__(self) -> None:
        super().__init__()

    def avf(self) -> None:
        pass

    def knn(self) -> None:
        pass

    def removeOutliers(self, detection_method: Callable) -> None:
        pass
