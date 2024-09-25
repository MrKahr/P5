import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

# Remove rows where value could not be determined
df.drop(df[(df["Ødem"] == 2) | (df["Ødem"] == 100)].index, inplace = True) # TODO: remove 100 during data cleaning

# Plot data over time
Plotter.groupedBarPlot(Plotter(), df, "Ødem")