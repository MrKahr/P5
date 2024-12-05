import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.reportContent.dataExploration.visualization import Plotter


dp = DataPreprocessor(Dataset.MÅL)
cleaner = DataCleaner(dp.df)
cleaner.cleanMålDataset()
df = dp.df

p = Plotter()
p.scatterPlot(
    dataFrame=df, attribute_x="Sårrand (cm)", attribute_y="Midte (cm)", colName="Dag"
)
