# Main entry-point into our code base
import os
import sys
from pathlib import Path

from modules.modelSelection.model_selection import ModelSelector

##########################
### Initial Path Setup ###
##########################
# Set initial CWD
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.transformer import DataTransformer

# dp = DataPreprocessor(Dataset.REGS)
# cleaner = DataCleaner(dp.df)
# cleaner.cleanRegsDataset()
# cleaner.deleteNonfeatures()
# dp.df = cleaner.getDataframe()
# fs = FeatureSelector(None, None)

# X, y = dp.getTrainingData(), dp.getTargetData()
# x_labels = dp.getTrainingLabels()
# fs.genericUnivariateSelect(
#     X=X,
#     y=y,
#     scoreFunc=fs._mutualInfoClassif,
#     mode="percentile",
#     param=50,
#     x_labels=x_labels,
# )

model = ModelSelector.getModel()
print(model)
