# Libraries 
import pandas as pd # CSV-reading 
import numpy as np # Numeric/Scientific/Math computation 
import matplotlib as mpl # Plotting 
from pathlib import Path
import os


class DataProcessor():
    def __init__(self, path: str) -> None:
        print(path)
        print(os.path.splitext(__file__)[0])
        print(Path(os.path.splitext(__file__)[0]).parents[1])
        self.data = pd.read_excel(path) # Read excel file as dataframe  - Use pandas to get descriptive statistics
        self.data = pd.DataFrame.to_numpy(self.data) # Change to numpy data format 
        print(self.data)
    
    def deleteCols(self, a, b) -> None:
        self.data = np.delete(self.data, [a,b])
    
    






dp = DataProcessor(Path(Path(os.path.split(__file__)[0]).parents[1], "data/Eksperimentelle_sår_2014.xlsx"))