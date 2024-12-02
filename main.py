#####################
### Initial Setup ###
#####################
import os

# Set initial CWD
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from modules.tools.arguments.app_arguments import AppArguments

# Setup CLI
arguments = AppArguments()
arguments.executeArguments()

######################
### Module Imports ###
######################
from modules.config.utils.setup_config import SetupConfig
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.pipeline import Pipeline


if SetupConfig.arg_batch:

    pass
else:
    Pipeline(Dataset.REGS).run()
