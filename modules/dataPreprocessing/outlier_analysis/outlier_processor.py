from typing import Callable

import pandas as pd

from KNN import KNNAnalysis
from AVF import AVFAnalysis


class OutlierProcessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def avf(self) -> list:
        classifier = AVFAnalysis(self.df)
        return classifier.calculateAVF()

    def knn(self) -> None:
        classifier = KNNAnalysis()

    def removeOutliers(self, detection_method: Callable) -> None:
        outliers = detection_method()
        self.df.drop(self.df[outliers.index])
