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

from modules.dataPreprocessing import DataProcessor, Dataset

class KNNAnalysis:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        dp.oneHotEncoding(["Eksudattype", "Hyperæmi"])
        self.df = dp.getDataFrame()
        self.df.drop(["Gris ID", "Sår ID"], axis = 1, inplace = True)
    
    def KNN(self,degree: int) -> None: 
        neighbourModel = NearestNeighbors(n_neighbors=degree)
        neighbourModel.fit(self.df)
        self.distances, self.neighbours = neighbourModel.kneighbors()

    def getIndegrees(self):
        """Generate dict storing indegree of every point with their key as index
        """
        indegrees = {}
        for point in self.neighbours:
            for neighbour in point:
                if not indegrees.get(neighbour): indegrees[neighbour] = 1
                else: indegrees[neighbour] = indegrees[neighbour] + 1
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


if __name__ == "__main__":
    op = KNNAnalysis()
    threshold = 0
    k = 30
    outliers = op.getOutliers(k, threshold)
    indegrees = op.getIndegrees()
    for i in range(len(op.df)):
        if not indegrees.get(i):
            indegrees[i] = 0
    n, bins, patches = plt.hist(indegrees.values(), bins=40)
    for i in range(threshold + 1):
        patches[i].set_color("r")
    plt.xlabel("Indegree")
    plt.ylabel("Frequency")
    plt.suptitle(f"Indegree distribution of {k} neighbors with threshold {threshold}")
    plt.show()
