#####################
### Initial Setup ###
#####################
import os
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
    # Batch processing
    config_list = list(
        ConfigBatchProcessor.getConfigPairsFromBatch(
            ConfigBatchProcessor.getBatchConfigs(SetupConfig.arg_batch_config_path)
        )
    )

    logger.info(f"Running in batch mode")
    with logging_redirect_tqdm(loggers=[logger]):
        for configs in tqdm(
            config_list,
            desc="Progress",
            unit="model",
            dynamic_ncols=True,
            colour="green",
            bar_format="{l_bar}{bar:15}{r_bar}",
        ):
            ConfigBatchProcessor.applyConfigs(configs)
            Pipeline(Dataset.REGS).run()
else:
    # Single processing
    Pipeline(Dataset.REGS).run()
