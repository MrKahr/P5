# Libraries
from enum import Enum
import os
from pathlib import Path
import pandas as pd  # CSV-reading, data manipulation and cleaning.


class DataType(Enum):
    MÅL = "eksperimentelle_sår_2024_mål.csv"
    REGS = "eksperimentelle_sår_2024_regs.csv"
    OLD = "old_eksperimentelle_sår_2014_regs"


class DataProcessor:
    def __init__(self, type: DataType) -> None:
        path = Path(
            Path(os.path.split(__file__)[0]).parents[0],
            f"data/{type.value}",
        )
        self.dataFrame = pd.read_csv(path, sep=";", comment="#")
        self.deleteNaN()

    def showDataFrame(self) -> None:
        print(self.dataFrame)

    def deleteNaN(self) -> None:
        self.dataFrame.dropna(axis=1, how="all", inplace=True)  # Columns
        self.dataFrame.dropna(axis=0, how="any", inplace=True)  # Rows

    def getCol(self, indices: list[str]) -> pd.DataFrame:
        tempdf = self.dataFrame[indices]
        print(tempdf)
        return tempdf

    def generateDecriptiveStats(self) -> pd.DataFrame:
        return self.dataFrame.describe()
