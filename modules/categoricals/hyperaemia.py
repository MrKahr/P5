import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter


class HyperaemiaPlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()

    def plotHyperaemia(self) -> None:
        p = Plotter()
        p.stackedBarPlot(dataframe=self.df, attribute_x="Dag", attribute_y="Hyperæmi")


if __name__ == "__main__":
    op = HyperaemiaPlotter()
    op.plotHyperaemia()
