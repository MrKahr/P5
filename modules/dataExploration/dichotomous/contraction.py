import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter

# Process data
dp = DataPreprocessor(Dataset.REGS)
cleaner = DataCleaner(dp.df)
cleaner.cleanRegs()
df = dp.df
df.drop(df[(df["Kontraktion"] == 100)].index,inplace=True)


# Plot data over time
Plotter().groupedBarPlot(
    df,
    "Dag",
    "Kontraktion",
    show_percentage=False,
    labels=["nej", "ja", "kan ikke vurderes", "100"],
)
