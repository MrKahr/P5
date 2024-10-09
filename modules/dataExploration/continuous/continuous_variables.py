# Include modules for barplot, historgram etc
import sys
import os

sys.path.insert(0, os.getcwd())

from numpy.typing import ArrayLike
import matplotlib.pyplot as plt  # Plotting


from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter

plt.close("all")  # closes all currently active figures


class ContinuousPlotter:
    def __init__(self) -> None:
        dp = DataPreprocessor(Dataset.MÅL)
        cleaner = DataCleaner(dp.df)
        cleaner.cleanMålDataset()
        self.df = dp.df

    def plotContinuous(self) -> None:
        p = Plotter()
        p.plotContinuous(dataFrame=self.df, y1="Sårrand (cm)", y2="Midte (cm)", x="Dag")


if __name__ == "__main__":
    op = ContinuousPlotter()
    op.plotContinuous()
