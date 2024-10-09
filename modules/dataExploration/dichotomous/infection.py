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
cleaner.deleteMissingValues()
df = dp.df

# Note: the dataset lists possible values as 1, 2 or 3, but only 1 and 2 are actually used

# Plot data over time
Plotter().groupedBarPlot(
    df,
    "Dag",
    "Infektionsniveau",
    show_percentage=False,
    labels=["rent", "kontamineret", "manglende værdi"],
)
