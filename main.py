##########################
### Initial Path Setup ###
##########################
# Set initial CWD
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

######################
### Module Imports ###
######################

from modules.pipeline import Pipeline
from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.transformer import DataTransformer

pipeline = Pipeline(Dataset.REGS)
pipeline.run()
