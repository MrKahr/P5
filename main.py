#####################
### Initial Setup ###
#####################
import os
from tqdm import tqdm

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
from modules.config.utils.config_batch_processor import ConfigBatchProcessor
from modules.logging import logger

if SetupConfig.arg_batch:
    config_list = ConfigBatchProcessor.getConfigPairsFromBatch(
        ConfigBatchProcessor.getBatchConfigs(SetupConfig.config_dir)
    )
    logger.info(f"Running in batch mode. Processing {len(config_list)} experiments")
    for configs in tqdm(config_list):
        ConfigBatchProcessor.applyConfigs(configs)
        Pipeline(Dataset.REGS).run()
else:
    Pipeline(Dataset.REGS).run()
