import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataProcessor, Dataset
from modules.dataExploration.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.df


# Plot data over time
Plotter().groupedBarPlot(
    df, "Dag", "Granulationsvæv", show_percentage=True, labels=["nej", "ja"]
)
