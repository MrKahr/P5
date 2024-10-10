import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter

"""
0: nej
1: omkring sår
2: i sår
3: kan ikke vurderes
4: omkring + i sår

"""


class HyperaemiaPlotter:
    def __init__(self) -> None:
        dp = DataPreprocessor(Dataset.REGS)
        cleaner = DataCleaner(dp.df)
        cleaner.cleanRegsDataset()
        self.df = dp.df

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
