import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

# Note: the dataset lists possible values as 1, 2 or 3, but only 1 and 2 are actually used

# Plot data over time
Plotter.groupedBarPlot(Plotter(), df, "Infektionsniveau")