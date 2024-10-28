import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

sys.path.insert(0, os.getcwd())

from modules.config.config_enums import OutlierRemovalMethod
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.pipeline import Pipeline
from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.transformer import DataTransformer
from modules.logging import logger


class AVFAnalysis:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        try:
            self.df.drop("Gris ID", axis=1, inplace=True)
        except KeyError:
            logger.info("Tried to remove 'Gris ID', but feature was already removed")
        try:
            self.df.drop("Sår ID", axis=1, inplace=True)
        except KeyError:
            logger.info("Tried to remove 'Sår ID', but feature was already removed")

    def AVF(self, row: dict) -> float:
        """Calculate AVF score for a row

        Parameters
        ----------
        row : dict
            Generated from iterrows() on a DataFrame

        Returns
        -------
        float
            AVF score
        """
        sum = 0
        keys = row.keys()
        i = 0
        for attributeVal in row:
            sum += self.frequency(attributeVal, keys[i])
            i += 1
        return (1 / len(row)) * sum

    def frequency(self, value: any, attribute: str, frequencies={}) -> float:
        """Calculate frequency of values in the dataset and store it for future calls to frequency()

        Parameters
        ----------
        value : any
            A value present in our dataset
        attribute : str
            The name of the attribute/feature containing the value
        frequencies : dict, optional
            Dict to store the frequencies for future calls, by default {}. When the function is called with default value, frequencies is only initialized once, and read on future calls

        Returns
        -------
        float
            The number of times the value appears in the DataFrame
        """

        # frequencies is only initialized once
        if not frequencies.get(attribute):
            frequencies[attribute] = self.valueOccurences(attribute)
        return frequencies[attribute][value]

    def valueOccurences(self, attribute: str) -> dict:
        """Calculates the number of times each value appears for an attribute and stores this in a dict

        Parameters
        ----------
        attribute : str
            Name of an attribute/feature/value

        Returns
        -------
        dict
            A dict containing keys of all attributes and values for how many times they occur
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

    def getOutliers(self, k: int) -> list:
        """Generates list of k outliers using AVF

        Parameters
        ----------
        k : int
            number of outliers

        Returns
        -------
        list
            list of outlier indexes
        """
        scores = self.calculateAVF()
        # https://www.geeksforgeeks.org/python-find-the-indices-for-k-smallest-elements/
        indexes = sorted(range(len(scores)), key=lambda sub: scores[sub])[:k]
        return indexes

    def plotAVFs(self, cutoffPercentile: float) -> None:
        """Plots outliers based on cutoffPercentile
        e.g. for 0.01, the lowest 1% AVFs scores determine the outlier cutoff bin

        Parameters
        ----------
        cutoffPercentile : float
            Lower percentile to consider as outliers
        """
        listAVF = self.calculateAVF()

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
    config.setValue("OutlierRemovalMethod", OutlierRemovalMethod.AVF.name)

    avf_param_key = "avfParams"
    config.setValue("k", 10, avf_param_key)

    # FIXME: Data loading in pipeline causing circular import!
    df = DataCleaner(Pipeline.loadDataset(dataset), dataset).run()
    transformer = DataTransformer(df)
    transformer.oneHotEncode(["Eksudattype", "Hyperæmi"])

    op = AVFAnalysis(transformer.df)
    op.plotAVFs(0.01)
