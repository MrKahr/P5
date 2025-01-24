import sys
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import OutlierRemovalMethod
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.transformer import DataTransformer


class KNNAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Perform k-nearest neighbors analysis of `df`.

        Parameters
        ----------
        df : dict
            DataFrame of features to perform KNN analysis on.
        """
        # Ensure that Gris ID and Sår ID are removed as they're useless for outlier analysis
        df.drop(["Gris ID", "Sår ID"], axis=1, inplace=True, errors="ignore")
        self.df = df

    def knn(self, degree: int) -> None:
        """
        Generate a k-nearest neighbors graph and
        store it in this class's instance variables `self.distances` and `self.neighbors`.

        Parameters
        ----------
        degree : int
            Number of neighbors.
        """
        neighborModel = NearestNeighbors(n_neighbors=degree)
        neighborModel.fit(self.df)
        self.distances, self.neighbors = neighborModel.kneighbors()

    def getIndegrees(self) -> dict:
        """
        Calculate indegrees for all points in `self.neighbors`.

        Returns
        -------
        dict
            Keys of point indexes and values of the point's indegree.
        """
        indegrees = {}
        for point in self.neighbors:
            for neighbor in point:
                if not indegrees.get(neighbor):
                    indegrees[neighbor] = 1
                else:
                    indegrees[neighbor] = indegrees[neighbor] + 1
        # print(indegrees)
        return indegrees

    def getOutliers(self, k: int, T: int) -> list[int]:
        """
        ODIN algorithm for outlier detection using k nearest neighbors with threshold T.

        Based on https://ieeexplore-ieee-org.zorac.aub.aau.dk/stamp/stamp.jsp?tp=&arnumber=1334558

        Parameters
        ----------
        k : int
            Number of neighbors.
        T : int
            Indegree threshold for when a point is considered an outlier.

        Returns
        -------
        list[int]
            Outlier indices in the dataframe.
        """
        self.knn(k)
        indegrees = self.getIndegrees()

        outliers = []
        for i in range(len(self.df)):
            if (not indegrees.get(i)) or indegrees[i] <= T:
                outliers.append(i)
        return outliers

    def plotNeighborMultiHist(
        self, nearest_neighbors: list | int, threshold=0, rows=3, cols=1
    ) -> None:
        """
        Plot histogram of indegrees and mark outliers in red.

        Parameters
        ----------
        nearest_neighbors : list | int
            One value for k neighbours in each graph.

        threshold : int, optional
            Max indegree to consider outlier, by default 0.

        rows : int, optional
            Number of rows for the subplots, by default 3.

        cols : int, optional
            Number of columns for the subplots, by default 1.

        Raises
        ------
        AssertionError
            If number of plots cannot fit in the given number of rows and columns.
        """
        # Ensure plot dimension and `nearest_neighbors` are a valid combination
        assert rows * cols != len(
            nearest_neighbors
        ), f"Cannot plot {rows * cols} plots with only {len(nearest_neighbors)} nearest neighbors"

        # Plot multiple historgrams for different number of columns and rows
        fig, axes = plt.subplots(nrows=rows, ncols=cols, layout="constrained")
        if rows == 1 and cols == 1:
            self.getOutliers(nearest_neighbors[0], threshold)
            indegrees = self.getIndegrees()
            for i in range(len(self.df)):
                if not indegrees.get(i):
                    indegrees[i] = 0
            n, bins, patches = axes.hist(indegrees.values(), bins=60)
            for i in range(threshold + 1):
                patches[i].set_color("r")
            axes.set_title(
                f"Indegree distribution of {nearest_neighbors[0]} neighbours with threshold {threshold}"
            )
        elif cols == 1:
            for indexRow in range(0, len(nearest_neighbors)):
                self.getOutliers(nearest_neighbors[indexRow], threshold)
                indegrees = self.getIndegrees()
                for i in range(len(self.df)):
                    if not indegrees.get(i):
                        indegrees[i] = 0
                n, bins, patches = axes[indexRow].hist(indegrees.values(), bins=60)
                for i in range(threshold + 1):
                    patches[i].set_color("r")
                axes[indexRow].set_title(
                    f"Indegree distribution of {nearest_neighbors[indexRow]} neighbours with threshold {threshold}"
                )
        else:
            k = 0
            for indexCol in range(cols):
                for indexRow in range(rows):
                    self.getOutliers(nearest_neighbors[k], threshold)
                    indegrees = self.getIndegrees()
                    for i in range(len(self.df)):
                        if not indegrees.get(i):
                            indegrees[i] = 0
                    n, bins, patches = axes[indexRow, indexCol].hist(
                        indegrees.values(), bins=60
                    )
                    for i in range(threshold + 1):
                        patches[i].set_color("r")
                    axes[indexRow, indexCol].set_title(
                        f"Indegree distribution of {nearest_neighbors[k]} neighbours with threshold {threshold}"
                    )
                    k += 1
        plt.show()


if __name__ == "__main__":
    # Testing code to check if KNN works
    from modules.pipeline import Pipeline

    dataset = Dataset.REGS
    config = PipelineConfig()

    general_key = "General"
    config.setValue("UseCleaner", True, general_key)
    config.setValue("UseTransformer", True, general_key)

    cleaning_key = "Cleaning"
    config.setValue("DeleteNanColumns", True, cleaning_key)
    config.setValue("DeleteNonfeatures", True, cleaning_key)
    config.setValue("DeleteMissingValues", True, cleaning_key)
    config.setValue("RemoveFeaturelessRows", True, cleaning_key)
    config.setValue("OutlierRemovalMethod", OutlierRemovalMethod.ODIN.name)

    odin_param_key = "odinParams"
    config.setValue("k", 30, odin_param_key)
    config.setValue("T", 0, odin_param_key)

    df = DataCleaner(Pipeline.loadDataset(dataset), dataset).run()
    transformer = DataTransformer(df)
    transformer.oneHotEncode(["Eksudattype", "Hyperæmi"])

    op = KNNAnalysis(transformer.df)
    threshold = 0
    k = [10, 20, 30]
    op.plotNeighborMultiHist(k, threshold, 3, 1)
