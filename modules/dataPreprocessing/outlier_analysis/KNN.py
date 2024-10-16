import sys
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.transformer import DataTransformer


class KNNAnalysis:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df=df
        if "Gris ID" in self.df:
            self.df.drop("Gris ID", axis=1, inplace=True)
        if "Sår ID" in self.df:
            self.df.drop("Sår ID", axis=1, inplace=True)
        
        transformer = DataTransformer(self.df)
        transformer.minMaxNormalization("Dag")
        self.df = transformer.getDataframe()
        

    def KNN(self, degree: int) -> None:
        """Generate a k nearest neighbors graph and store it in the KNN class as ndarrays self.distances and self.neighbors

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


if __name__ == "__main__":
    dp = DataPreprocessor(Dataset.REGS)

    cleaner = DataCleaner(dp.df)
    cleaner.cleanRegsDataset()
    cleaner.deleteMissingValues()

    transformer = DataTransformer(cleaner.getDataframe())
    transformer.oneHotEncode(["Eksudattype", "Hyperæmi"])

    op = KNNAnalysis(transformer.getDataframe())
    threshold = 0
    k = [10, 20, 30]
    op.PlotNeighborMultiHist(k, 0, 3, 1)