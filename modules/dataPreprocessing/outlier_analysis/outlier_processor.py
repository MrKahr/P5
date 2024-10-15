import sys
import os

sys.path.insert(0, os.getcwd())

import pandas as pd

from modules.dataPreprocessing.outlier_analysis.KNN import KNNAnalysis
from modules.dataPreprocessing.outlier_analysis.AVF import AVFAnalysis
from modules.logging import logger


class OutlierProcessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def avf(self, k: int) -> list:
        classifier = AVFAnalysis(self.df.copy(deep=False)) # Shallow copy used so ID can be removed for outlier detection, without affecting dataframe in self.df
        return classifier.getOutliers(k)

    def odin(self, k: int, T: int) -> list:
        classifier = KNNAnalysis(self.df.copy(deep=False))
        return classifier.getOutliers(k, T)

    def removeOutliers(self, outliers: list) -> None:
        """Remove outliers based on their index

        Parameters
        ----------
        outliers : list
            List of indexes of outliers. Can be generated by calling knn or avf
        """
        current_len = len(self.df)
        self.df.drop(self.df.iloc[outliers].index, inplace=True) # DataFrame is indexed in this way, as the index used by drop can differ from the one used by iloc
        amount = current_len - len(self.df)
        logger.info(f"Removed {amount} {"outliers" if amount != 1 else "outlier"}")