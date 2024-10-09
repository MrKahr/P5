import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter

# Process data
dp = DataPreprocessor(Dataset.REGS)
cleaner = DataCleaner(dp.df)
cleaner.cleanRegsDataset()
cleaner._deleteMissing()
df = dp.df


# Plot data over time
Plotter().groupedBarPlot(
    df,
    "Dag",
    "Eksudat",
    show_percentage=False,
    labels=["nej", "ja", "kan ikke vurderes"],
)
