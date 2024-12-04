#####################
### Initial Setup ###
#####################
from datetime import datetime
import os
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Set initial CWD
os.environ["KERAS_BACKEND"] = "torch"

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
from modules.modelSummary.export_summary import SummaryExporter
from modules.logging import logger

if SetupConfig.arg_batch:
    # Batch processing
    config_list = list(
        ConfigBatchProcessor.getConfigPairsFromBatch(
            ConfigBatchProcessor.getBatchConfigs(SetupConfig.arg_batch_config_path)
        )
    )
    total_batches = len(config_list) - 1
    batch_id = f"batch.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"

    logger.info(f"Running in batch mode")
    with logging_redirect_tqdm(loggers=[logger]):
        for i, configs in enumerate(
            tqdm(
                config_list,
                desc="Progress",
                unit="model",
                dynamic_ncols=True,
                colour="green",
                bar_format="{l_bar}{bar:15}{r_bar}",
            )
        ):
            ConfigBatchProcessor.applyConfigs(configs)
            pipeline_report = Pipeline(Dataset.REGS).run()
            SummaryExporter.export(pipeline_report, i, total_batches, batch_id)
else:
    # Single processing
    Pipeline(Dataset.REGS).run()
