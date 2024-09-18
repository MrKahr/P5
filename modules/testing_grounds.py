# Libraries
from pathlib import Path
import os
import pandas as pd  # CSV-reading, and data manipulation
import matplotlib.pyplot as plt  # plotting

plt.close("all")  # closes all currently active figures


class DataProcessor:
    def __init__(self, path: str) -> None:
        self.dataFrame = pd.read_csv(path, sep=";")

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

# Remove NaN from data
dp.deleteNaN(1)

# Get columns by index
subdf = dp.getCol(["Tid", "Ødem"])
print(subdf)

# Get discriptive stats (mean, quartiles, etc)
statsDf = dp.generateDecriptiveStats()
print(statsDf)

# Initialize new figure - see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
print(dp.dataFrame["Niveau sårvæv"])

plt.figure()
dp.dataFrame["Niveau sårvæv"].plot(
    kind="hist", bins=10, title="My first Plot", xlabel="X-AXSIS", ylabel="Y-AXSIS"
)
plt.show()
