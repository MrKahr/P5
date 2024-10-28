##########################
### Initial Path Setup ###
##########################
# Set initial CWD
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

######################
### Module Imports ###
######################

from modules.dataPreprocessing.dataset_enums import Dataset
from modules.pipeline import Pipeline

pipeline = Pipeline(Dataset.REGS, Dataset.OLD)
pipeline.run()
