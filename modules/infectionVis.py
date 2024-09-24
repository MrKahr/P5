from dataPreprocessing import DataProcessor, Dataset
import visualization

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

# Note: the dataset lists possible values as 1, 2 or 3, but only 1 and 2 are actually used

# Plot data over time
visualization.groupedBarPlot("Infektionsniveau", df)