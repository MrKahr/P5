import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter


dp = DataProcessor(Dataset.MÅL)
df = dp.df

p = Plotter()
p.scatterPlot(dataFrame=df, attribute_x="Sårrand (cm)", attribute_y="Midte (cm)", colName="Dag")