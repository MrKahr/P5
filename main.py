#####################
### Initial Setup ###
#####################
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Set initial CWD


####################################
### Setup for GPU-based training ###
####################################
# Set environ BEFORE any CUDA-related imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_PATH"] = str(Path(Path.cwd(), r".conda\Library"))

import torch
import torch._dynamo.config
import torch._dynamo.logging

torch.set_float32_matmul_precision("high")  # Use TF32 or approximate FP32
torch._dynamo.config.cache_size_limit = 16  # Double cache size for compilation
torch._dynamo.config.suppress_errors = True  # Prevent weird TypeError in Keras model


###################
### # Setup CLI ###
###################
from modules.tools.arguments.app_arguments import AppArguments

arguments = AppArguments()
arguments.executeArguments()

from modules.logging.logger import Logger

# Show start header after CLI is initialized
Logger.writeHeaderToLog()

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
    batch_id = f"batch.{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"

    logger.info(f"Running in batch mode")

    if not config_list:
        logger.warning(
            f"Found no configs at '{SetupConfig.arg_batch_config_path}'. Please export some using the CLI option '--export'"
        )
        exit(0)

    # Tell tqdm to consider logging module when printing progress bar
    with logging_redirect_tqdm(loggers=[logger, *torch._dynamo.logging.get_loggers()]):
        for i, configs in enumerate(
            tqdm(
                config_list,
                desc="Overall Progress ",
                unit="model",
                dynamic_ncols=True,
                colour="green",
                position=1,
                bar_format="{l_bar}{bar:50}{r_bar}",
            )
        ):
            ConfigBatchProcessor.applyConfigs(configs)
            pipeline_report = Pipeline(Dataset.REGS).run()
            SummaryExporter.export(
                pipeline_report,
                batch_id,
                os.path.splitext(os.path.split(SetupConfig.pipeline_config_path)[1])[0],
            )
            SummaryExporter.writeKeyToLatexTable(
                str(SetupConfig.pipeline_config_path),
                pipeline_report,
                "test_accuracies",
                Path("Experiments for the report", "accuracies"),
                "&",
            )
else:
    # Single processing
    pipeline_report = Pipeline(Dataset.REGS).run()
    SummaryExporter.export(
        pipeline_report, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    SummaryExporter.writeKeyToLatexTable(
        str(SetupConfig.pipeline_config_path),
        pipeline_report,
        "test_accuracies",
        Path("Experiments for the report", "accuracies"),
        "&",
    )
