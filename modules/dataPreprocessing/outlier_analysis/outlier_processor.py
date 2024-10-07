from typing import Callable

import pandas as pd


class OutlierProcessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def avf(self) -> None:
        pass

    def knn(self) -> None:
        pass

    def removeOutliers(self, detection_method: Callable) -> None:
        pass
