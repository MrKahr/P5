import sys
import os

sys.path.insert(0, os.getcwd())

import pandas as pd

from modules.dataPreprocessing.outlier_analysis.KNN import KNNAnalysis
from modules.dataPreprocessing.outlier_analysis.AVF import AVFAnalysis
from modules.logging import logger
from modules.config.config import Config
from modules.config.config_enums import OutlierRemovalMethod


class OutlierProcessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def avf(self, k: int) -> list:
        """Generates indexes of outliers using frequency of attribute values without modifying the DataFrame

        Parameters
        ----------
        k : int
            Number of outliers to generate

        Returns
        -------
        list
            Indexes of outliers in DataFrame
        """
        classifier = AVFAnalysis(
            self.df.copy(deep=False)
        )  # Shallow copy used so ID can be removed for outlier detection, without affecting dataframe in self.df
        return classifier.getOutliers(k)

    def odin(self, k: int, T: int) -> list:
        """Generates indexes of outliers using k-nearest-neighbors without modifying the DataFrame

        Parameters
        ----------
        k : int
            Number of neighbors
        T : int
            Indegree threshold

        Returns
        -------
        list
            Indexes of outliers in DataFrame
        """
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
        self.df.drop(
            self.df.iloc[outliers].index, inplace=True
        )  # DataFrame is indexed in this way, as the index used by drop can differ from the one used by iloc
        amount = current_len - len(self.df)
        logger.info(f"Removed {amount} {"outliers" if amount != 1 else "outlier"}")

    def getDataframe(self) -> pd.DataFrame:
        """Get the dataframe with outliers processed as a deep copy.
        Returns
        -------
        pd.DataFrame
            The dataframe with outliers processed
        """
        return self.df.copy(deep=True)

    def run(self) -> None:
        config = Config()
        match config.getValue("OutlierRemovalMethod"):
            case OutlierRemovalMethod.ODIN.name:
                self.odin(**config.getValue("odinParams"))
            case OutlierRemovalMethod.AVF.name:
                self.avf(**config.getValue("avfParams"))
            case OutlierRemovalMethod.NONE.name:
                logger.info("Removing no outliers.")
            case _:
                logger.warning(
                    "Undefined outlier removal method selected for OutlierProcessor! Removing no outliers."
                )
        logger.info("OutlierProcessor is done")
        return self.getDataframe()
