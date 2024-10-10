import sys
import os

import pandas as pd

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.transformer import DataTransformer

import matplotlib.pyplot as plt
import math


class AVFAnalysis:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        

    def AVF(self, row) -> float:
        sum = 0
        keys = row.keys()
        i = 0
        for attributeVal in row:
            sum += self.frequency(attributeVal, keys[i])
            i += 1
        return (1 / len(row)) * sum

    def frequency(self, value, attribute, frequencies={}) -> float:
        # frequencies is only initialized once
        if not frequencies.get(attribute):
            frequencies[attribute] = self.valueOccurences(attribute)
        return frequencies[attribute][value]

    def valueOccurences(self, attribute: str) -> dict:
        """Calculates the number of times each value appears for an attribute and stores this in a dict"""
        column = self.df[attribute]
        values = column.unique()
        sums = {}
        for i in values:
            sums[i] = 0
        for i in column:
            sums[i] += 1
        # Use to get ratio
        # for i in values:
        #     sums[i] = sums[i] / len(column)
        return sums
    
    def calculateAVF(self) -> list:
        """Generates list of AVF scores for a dataframe

        -------
        returns list of AVF scores
        """
        listAVF = []
        for index, row in self.df.iterrows():
            AVFelem = self.AVF(row)
            listAVF.append(AVFelem)
        return listAVF

    def plotAVFs(self, cutoffPercentile: float) -> None:
        """Plots outliers based on cutoffPercentile
        e.g. for 0.01, the lowest 1% AVFs scores determine the outlier cutoff bin
        """
        listAVF= self.calculateAVF()

        lowestPercentage = math.floor(len(listAVF) * cutoffPercentile)

        n, bins, patches = plt.hist(listAVF, bins=40)
        sumBars = 0
        i = 0
        while sumBars <= lowestPercentage:
            patches[i].set_color("r")
            sumBars += n[i]
            i += 1
        # print("length:", len(listAVF))
        plt.xlabel("AVF score")
        plt.ylabel("Datapoints in range")
        plt.suptitle("AVF score distribution")
        plt.show()


if __name__ == "__main__":
    dp = DataPreprocessor(Dataset.REGS)

    cleaner = DataCleaner(dp.df)
    cleaner.cleanRegsDataset()
    cleaner.deleteMissingValues()

    transformer = DataTransformer(dp.df)
    transformer.oneHotEncode(["Eksudattype", "Hyperæmi"])

    op = AVFAnalysis(cleaner.getDataframe())
    op.df.drop(["Gris ID", "Sår ID"], axis=1, inplace=True)
    op.plotAVFs(0.01)
