import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter

"""
0: nej
1: omkring sår
2: i sår
3: kan ikke vurderes
4: omkring + i sår

"""


class HyperaemiaPlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()

    def plotHyperaemia(self) -> None:
        p = Plotter()
        p.stackedBarPlot(
            dataframe=self.df,
            attribute_x="Dag",
            attribute_y="Hyperæmi",
            show_percentage=False,
            labels=[
                "nej",
                "omkring sår",
                "i sår",
                "kan ikke vurderes",
                "omkring + i sår",
            ],
        )


if __name__ == "__main__":
    op = HyperaemiaPlotter()
    op.plotHyperaemia()
