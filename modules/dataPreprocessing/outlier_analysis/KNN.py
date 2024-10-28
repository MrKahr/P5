import sys
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from modules.config.config import Config
from modules.config.config_enums import OutlierRemovalMethod
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.pipeline import Pipeline
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.transformer import DataTransformer


class KNNAnalysis:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        if "Gris ID" in self.df:
            self.df.drop("Gris ID", axis=1, inplace=True)
        if "Sår ID" in self.df:
            self.df.drop("Sår ID", axis=1, inplace=True)

        transformer = DataTransformer(self.df)
        transformer.minMaxNormalization("Dag")
        self.df = transformer.getDataframe()

    def KNN(self, degree: int) -> None:
        """Generate a k-nearest neighbors graph and store it in the KNN class as ndarrays self.distances and self.neighbors

        Parameters
        ----------
        degree : int
            number of neighbors
        """
        neighborModel = NearestNeighbors(n_neighbors=degree)
        neighborModel.fit(self.df)
        self.distances, self.neighbors = neighborModel.kneighbors()

    def getIndegrees(self) -> dict:
        """Calculate indegrees for all points in self.neighbors

        Returns
        -------
        dict
            Keys of point indexes and values of the point's indegree
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

    def getOutliers(self, k: int, T: int) -> list:
        """ODIN algorithm for outlier detection using k nearest neighbors with threshold T

        Based on https://ieeexplore-ieee-org.zorac.aub.aau.dk/stamp/stamp.jsp?tp=&arnumber=1334558

        Parameters
        ----------
        k : int
            number of neighbors
        T : int
            indegree threshold for when a point is considered an outlier

        Returns
        -------
        list
            list of outlier indexes
        """
        self.KNN(k)
        indegrees = self.getIndegrees()

        outliers = []
        for i in range(len(self.df)):
            if (not indegrees.get(i)) or indegrees[i] <= T:
                outliers.append(i)
        return outliers

    def PlotNeighborMultiHist(
        self, nearestNeighbors: list | int, threshold=0, rows=3, cols=1
    ) -> None:
        """Plot histogram of indegrees and mark outliers in red

        Parameters
        ----------
        nearestNeighbors : list | int
            One value for k neighbours in each graph
        threshold : int, optional
            Max indegree to consider outlier, by default 0
        rows : int, optional
            Number of rows for the subplots, by default 3
        cols : int, optional
            Number of columns for the subplots, by default 1

        Raises
        ------
        Exception
            Exception raised if number of plots cannot fit in the given number of rows and columns
        """
        # Plot multiple historgrams for different number of columns and rows
        if rows * cols != len(nearestNeighbors):
            raise Exception(
                f"Cannot plot {rows * cols} plots with only {len(nearestNeighbors)} k's"
            )
        fig, axes = plt.subplots(nrows=rows, ncols=cols, layout="constrained")
        if rows == 1 and cols == 1:
            self.getOutliers(nearestNeighbors[0], threshold)
            indegrees = self.getIndegrees()
            for i in range(len(self.df)):
                if not indegrees.get(i):
                    indegrees[i] = 0
            n, bins, patches = axes.hist(indegrees.values(), bins=60)
            for i in range(threshold + 1):
                patches[i].set_color("r")
            axes.set_title(
                f"Indegree distribution of {nearestNeighbors[0]} neighbours with threshold {threshold}"
            )
        elif cols == 1:
            for indexRow in range(0, len(nearestNeighbors)):
                self.getOutliers(nearestNeighbors[indexRow], threshold)
                indegrees = self.getIndegrees()
                for i in range(len(self.df)):
                    if not indegrees.get(i):
                        indegrees[i] = 0
                n, bins, patches = axes[indexRow].hist(indegrees.values(), bins=60)
                for i in range(threshold + 1):
                    patches[i].set_color("r")
                axes[indexRow].set_title(
                    f"Indegree distribution of {nearestNeighbors[indexRow]} neighbours with threshold {threshold}"
                )
        else:
            k = 0
            for indexCol in range(cols):
                for indexRow in range(rows):
                    self.getOutliers(nearestNeighbors[k], threshold)
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
                        f"Indegree distribution of {nearestNeighbors[k]} neighbours with threshold {threshold}"
                    )
                    k += 1
        plt.show()


# TODO: Include outlier analysis in model report for plotting (soon to be: pipeline report)
if __name__ == "__main__":
    dataset = Dataset.REGS
    config = Config()

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

    # FIXME: Data loading in pipeline causing circular import!
    df = DataCleaner(Pipeline.loadDataset(dataset), dataset).run()
    transformer = DataTransformer(df)
    transformer.oneHotEncode(["Eksudattype", "Hyperæmi"])

    op = KNNAnalysis(transformer.df)
    threshold = 0
    k = [10, 20, 30]
    op.PlotNeighborMultiHist(k, threshold, 3, 1)
