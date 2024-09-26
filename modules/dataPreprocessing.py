# Libraries
from enum import Enum
import os
from pathlib import Path
import pandas as pd  # CSV-reading, data manipulation and cleaning.


class Dataset(Enum):
    MÅL = "eksperimentelle_sår_2024_mål.csv"
    REGS = "eksperimentelle_sår_2024_regs.csv"
    OLD = "old_eksperimentelle_sår_2014_regs"


class DataProcessor:
    def __init__(self, type: Dataset) -> None:
        path = Path(
            Path(os.path.split(__file__)[0]).parents[0],
            f"data/{type.value}",
        )
        self.dataFrame = pd.read_csv(path, sep=";", comment="#")
        if(type == Dataset.REGS) : self.deleteNaN()

    def showDataFrame(self) -> None:
        print(self.dataFrame)

    def deleteNaN(self) -> None:
        self.dataFrame.dropna(axis=1, how="all", inplace=True)  # Columns

        # FIXME: Do NOT remove any rows with NaN!!!
        self.dataFrame.dropna(axis=0, how="any", inplace=True)  # Rows

    def getDataFrame(self) -> pd.DataFrame:
        return self.dataFrame
