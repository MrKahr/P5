import matplotlib.pyplot as plt  # Plotting
import numpy as np
import pandas as pd

from dataPreprocessing import DataProcessor, Dataset
from visualization import AccuracyPlotter


class OrdinalPlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()

    def plotScab(self) -> None:
        pass

    def plotWoundTissueLevel(self) -> None:
        # TODO: Reorder ordinals to have a 'natural' progression
        x = self.df["Niveau sårvæv"]


if __name__ == "__main__":
    op = OrdinalPlotter()
    op.plotScab()
