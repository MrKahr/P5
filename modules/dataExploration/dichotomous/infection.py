import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataProcessor, Dataset
from modules.dataExploration.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)
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
