# Distance between variables (is minkowski appropriate for categorical variables)?
# Should one hot coding be applied before distances are found?
# What should the k-nearest neighbours threshhold be?
import sys
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataProcessor, Dataset


class KNNAnalysis:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        dp.oneHotEncoding(["Eksudattype", "Hyperæmi"])
        self.df = dp.getDataFrame()
        self.df.drop(["Gris ID", "Sår ID"], axis=1, inplace=True)

    def KNN(self, degree: int) -> None:
        neighbourModel = NearestNeighbors(n_neighbors=degree)
        neighbourModel.fit(self.df)
        self.distances, self.neighbours = neighbourModel.kneighbors()

    def getIndegrees(self):
        """Generate dict storing indegree of every point with their key as index"""
        indegrees = {}
        for point in self.neighbours:
            for neighbour in point:
                if not indegrees.get(neighbour):
                    indegrees[neighbour] = 1
                else:
                    indegrees[neighbour] = indegrees[neighbour] + 1
        # print(indegrees)
        return indegrees

    def getOutliers(self, k: int, T: int) -> list:
        """ODIN algorithm for outlier detection using k nearest neighbors with threshold T

        Based on https://ieeexplore-ieee-org.zorac.aub.aau.dk/stamp/stamp.jsp?tp=&arnumber=1334558
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
        # Plot multiple historgrams for different number of columns and rows
        if rows * cols != len(nearestNeighbors):
            raise Exception(
                f"Cannot plot {rows * cols} plots with only {len(nearestNeighbors)}k's"
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
    op = KNNAnalysis()
    threshold = 0
    k = [10, 20, 30, 40]

    op.PlotNeighborMultiHist(k, 0, 2, 2)
