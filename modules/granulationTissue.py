# Main entry-point into our code base
import os
from pathlib import Path

from dataPreprocessing import DataProcessor
import visualization

# Process data
df = DataProcessor(
    Path(
        Path(os.path.split(__file__)[0]),
        "../data/eksperimentelle_sår_2024_regs.csv",
    )
).dataFrame

# Remove rows where value could not be determined
# https://stackoverflow.com/questions/13851535/how-to-delete-rows-from-a-pandas-dataframe-based-on-a-conditional-expression
df.drop(df[df["Granulationsvæv"] == 2].index, inplace = True)


visualization.groupedBarPlot("Granulationsvæv", df)