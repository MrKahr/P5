from pathlib import Path

import pandas as pd
from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.transformer import DataTransformer
from modules.dataPreprocessing.outlier_analysis import OutlierProcessor
from modules.modelSelection.model_selection import ModelSelector
from modules.modelTraining.model_training import ModelTrainer
from modules.modelTesting.model_testing import ModelTester
from modules.modelSummary.model_summarizing import ModelSummary
from numpy.typing import NDArray


from modules.logging import logger


class Pipeline:
    """This is the singleton responsible for running the entire pipeline
    # NOTE - Be very careful when making changes here"""

    def __init__(self, data: Dataset) -> None:
        logger.info(f"Loading '{data.name}' dataset")
        path = Path("data", data.value).absolute()
        self.df = pd.read_csv(path, sep=";", comment="#")
        self.model = None
        self.config = Config()

    def _formatTrainingData(self) -> pd.DataFrame:
        return self.df.drop(["Dag"], axis=1, inplace=False)

    def getTrainingData(self) -> NDArray:
        return self._formatTrainingData().to_numpy()

    def getTargetData(self) -> NDArray:
        return self.df["Dag"].to_numpy(copy=True)

    def getTrainingLabels(self) -> NDArray:
        return self._formatTrainingData().columns.values

    def getTargetMaxValue(self) -> int:
        ndarr = self.df["Dag"].unique()
        i = ndarr.argmax()
        return ndarr[i]

    def run(self) -> None:
        df = self.df
        if self.config.getValue("UseCleaner"):
            logger.info("Trying to run DataCleaner...")
            df = DataCleaner(df).run()
        if self.config.getValue("UseFeatureSelector"):
            logger.info("Trying to run FeatureSelector...")
            df = (
                FeatureSelector.run()
            )  # TODO figure out how to pass the dataset to FeatureSelector etc.
        if self.config.getValue("UseTransformer"):
            logger.info("Trying to run DataTransformer...")
            df = DataTransformer(df).run()
        if self.config.getValue("UseOutlierRemoval"):
            df = OutlierProcessor(
                df
            ).run()  # NOTE It might be smart to run outlier analysis before imputation (which happens in the DataTransformer). Consider.
        if self.config.getValue("UseModelSelector"):
            logger.info("Trying to run ModelSelector...")
            self.model = ModelSelector.run()
        if self.config.getValue("UseModelTrainer"):
            logger.info("Trying to run ModelTrainer...")
            estimator = ModelTrainer.run(df)
        if self.config.getValue("UseModelTester"):
            logger.info("Trying to run ModelTester...")
            ModelTester.run(self.getTrainingData(), self.getTargetData(), estimator)
        if self.config.getValue("UseModelSummary"):
            logger.info("Trying to run ModelSummary...")
            ModelSummary.run()
