# Libraries
from pathlib import Path
import os
import pandas as pd  # CSV-reading, data manipulation and cleaning.
import matplotlib.pyplot as plt  # Plotting

plt.close("all")  # closes all currently active figures


class DataProcessor:
    def __init__(self, path: str) -> None:
        self.dataFrame = pd.read_csv(path, sep=";", comment="#")

    def process(self) -> None:
        self.deleteNaN(self, "Col")

    def showDataFrame(self) -> None:
        print(self.dataFrame)

    def deleteNaN(self, RowOrCol: int | str) -> None:
        self.dataFrame.dropna(axis=RowOrCol, how="all", inplace=True)

    def getCol(self, indices: list[str]) -> pd.DataFrame:
        tempdf = self.dataFrame[indices]
        print(tempdf)
        return tempdf

    def generateDecriptiveStats(self) -> pd.DataFrame:
        return self.dataFrame.describe()


# Load data
dp = DataProcessor(
    Path(
        Path(os.path.split(__file__)[0]).parents[0],
        "data/eksperimentelle_sår_2024_regs.csv",
    )
)
