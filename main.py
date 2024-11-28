#####################
### Initial Setup ###
#####################
# Set initial CWD
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


from modules.logging.logger import Logger
from modules.tools.arguments.app_arguments import AppArguments

arguments = AppArguments()
Logger.writeHeaderToLog()  # Prevent multi-threading bug with startup header
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
