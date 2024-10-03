import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataProcessor, Dataset
from modules.dataExploration.visualization import Plotter


class ScabPlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()

    def plotScab(self) -> None:
        p = Plotter()
        p.stackedBarPlot(
            dataframe=self.df,
            attribute_x="Dag",
            attribute_y="Sårskorpe",
            bar_width=1.5,
            labels=["nej", "delvist dækket", "ja fuldt dækket"],
        )


if __name__ == "__main__":
    op = ScabPlotter()
    op.plotScab()
