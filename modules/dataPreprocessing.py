# Libraries
import pandas as pd  # CSV-reading, data manipulation and cleaning.


class DataProcessor:
    def __init__(self, path: str) -> None:
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
