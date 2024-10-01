import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset

import pandas as pd
import matplotlib.pyplot as plt
import math


class AVFPlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()
        self.df.drop(["Gris ID", "Sår ID"], axis = 1, inplace = True)
        self.plotAVFs(self.df, 0.01)
    
    def AVF(self, row) -> float:
        sum = 0
        keys = row.keys()
        i = 0
        for attributeVal in row:
            sum += self.frequency(attributeVal, keys[i])
            i += 1
        return (1/len(row))*sum

    def frequency(self, value, attribute, frequencies={}) -> float:
        # frequencies is only initialized once
        if not frequencies.get(attribute):
            frequencies[attribute] = self.valueOccurences(attribute)
        return frequencies[attribute][value]

    def valueOccurences(self, attribute: str) -> dict:
        """Calculates the number of times each value appears for an attribute and stores this in a dict
        """
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

    def plotAVFs(self, df: pd.DataFrame, cutoffPercentile:float) -> None:
            """Plots outliers based on cutoffPercentile 
            e.g. for 0.01, the lowest 1% AVFs scores determine the outlier cutoff bin
            """
            listAVF = []
            sum = 0
            for index, row in df.iterrows():
                AVFelem = self.AVF(row)
                listAVF.append(AVFelem)
                sum += AVFelem
            avg = sum / len(listAVF)
            lowestPercentage = math.floor(len(listAVF) * cutoffPercentile)
            print("average:", avg)
            sortedList = sorted(listAVF)[:lowestPercentage]
            print("outliers:", sortedList)

            n, bins, patches = plt.hist(listAVF, bins=40)
            sumBars = 0
            i = 0
            while sumBars <= lowestPercentage:
                patches[i].set_color("r")
                sumBars += n[i]
                i += 1
            print("length:", len(listAVF))
            plt.xlabel("AVF score")
            plt.ylabel("Datapoints in range")
            plt.suptitle("AVF score distribution")
            plt.show()

if __name__ == "__main__":
    op = AVFPlotter()

