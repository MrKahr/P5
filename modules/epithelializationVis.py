from dataPreprocessing import DataProcessor, Dataset
import visualization

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

# Remove rows where value could not be determined
df.drop(df[df["Epithelialisering"] == 2].index, inplace = True)

# Plot data over time
visualization.groupedBarPlot("Epithelialisering", df)