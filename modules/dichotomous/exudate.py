import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

# Remove rows where value could not be determined
df.drop(df[df["Eksudat"] == 2].index, inplace = True)

# Plot data over time
Plotter.groupedBarPlot(Plotter(), df, "Eksudat")