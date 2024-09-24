from dataPreprocessing import DataProcessor, Dataset
import visualization

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

# Remove rows where value could not be determined
df.drop(df[df["Granulationsvæv"] == 2].index, inplace = True)

visualization.groupedBarPlot("Granulationsvæv", df)