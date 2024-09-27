import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.df


# Plot data over time
Plotter().groupedBarPlot(
    df, "Dag", "Eksudat", show_percentage=False, labels=["nej", "ja"]
)
