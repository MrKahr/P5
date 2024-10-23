from pathlib import Path

import pandas as pd
from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.transformer import DataTransformer
from modules.modelSelection.model_selection import ModelSelector
from modules.modelTraining.model_training import ModelTrainer
from modules.modelTesting.model_testing import ModelTester
from modules.modelSummary.model_summarizing import ModelSummary

from modules.logging import logger


class Pipeline:
    """This is the singleton responsible for running the entire pipeline
    # NOTE - Be very careful when making changes here"""

    def __init__(self, data: Dataset) -> None:
        logger.info(f"Loading '{data.name}' dataset")
        path = Path("data", data.value).absolute()
        self.df = pd.read_csv(path, sep=";", comment="#")
        self.config = Config()

    def run(self) -> None:
        if self.config.getValue("UseCleaner"):
            logger.info("Trying to run DataCleaner...")
            DataCleaner(self.df).run()
        if self.config.getValue("UseFeatureSelector"):
            logger.info("Trying to run FeatureSelector...")
            FeatureSelector().run()
        if self.config.getValue("UseTransformer"):
            logger.info("Trying to run DataTransformer...")
            DataTransformer().run()
        if self.config.getValue("UseModelSelector"):
            logger.info("Trying to run ModelSelector...")
            ModelSelector.run()
        if self.config.getValue("UseModelTrainer"):
            logger.info("Trying to run ModelTrainer...")
            ModelTrainer().run()
        if self.config.getValue("UseModelTester"):
            logger.info("Trying to run ModelTester...")
            ModelTester().run()
        if self.config.getValue("UseModelSummary"):
            logger.info("Trying to run ModelSummary...")
            ModelSummary().run()
