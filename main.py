##########################
### Initial Path Setup ###
##########################
# Set initial CWD
import os

from modules.dataPreprocessing.processor import Processor
from modules.dataPreprocessing.strategy_parse_config import StrategyparseConfig

os.chdir(os.path.dirname(os.path.abspath(__file__)))

######################
### Module Imports ###
######################

from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.transformer import DataTransformer

config = Config()
dataProcessor = DataPreprocessor(Dataset.REGS)
processor = Processor(
    config.getValue("Cleaning"),
    PipelineComponent=DataCleaner(dataProcessor.df),
    Strategy=StrategyparseConfig(),
)
processor.performAlgorithm()
