import pandas as pd
from pathlib import Path
from numpy.typing import NDArray

from modules.config.config import Config
from modules.crossValidationSelection.cross_validation_selection import (
    CrossValidationSelector,
)
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.outlier_analysis.outlier_processor import (
    OutlierProcessor,
)
from modules.dataPreprocessing.transformer import DataTransformer
from modules.logging import logger
from modules.modelSelection.model_selection import ModelSelector
from modules.modelTraining.model_training import ModelTrainer
from modules.modelTesting.model_testing import ModelTester
from modules.modelSummary.model_summarizing import ModelSummary


class Pipeline:
    _logger = logger
    _config = Config()

    def __init__(self, train_dataset: Dataset) -> None:
        """
        Responsible for running the entire pipeline, collecting datasets for each iteration
        and providing args to pipeline parts.

        # NOTE - Be very careful when making changes here.

        Parameters
        ----------
        train_dataset : Dataset
            Dataset used for model training.
            Please ensure it is compatible with `test_dataset`.
        """
        self.train_dataset = train_dataset
        self.df = None  # type: pd.DataFrame
        self.selected_features = None  # type: NDArray

    @classmethod
    def loadDataset(cls, dataset: Dataset) -> pd.DataFrame:
        logger.info(f"Loading '{dataset.name}' dataset")
        path = Path("dataset", dataset.value).absolute()
        return pd.read_csv(path, sep=";", comment="#")

    def getTrainX(self) -> pd.DataFrame:
        return self.df.drop(["Dag"], axis=1, inplace=False)

    def getTrueY(self) -> pd.Series:
        return self.df["Dag"]

    def run(self) -> None:
        """Using a dataframe, calls relevant pipeline components to perform transformations of dataset as specified in the config file"""
        self._logger.info(
            f"Initializing Pipeline: training dataset '{self.train_dataset.name}'"
        )
        self.df = DataCleaner(
            self.loadDataset(self.train_dataset), self.train_dataset
        ).run()
        self.df = OutlierProcessor(self.df).run()
        self.df = DataTransformer(self.df).run()

        # Unsplit is the dataset not split into train/test
        unsplit_x, unsplit_true_y, self.selected_features = FeatureSelector(
            self.getTrainX(), self.getTrueY()
        ).run()

        pipeline_report = ModelTrainer(
            estimator=ModelSelector.getModel(),
            cross_validator=CrossValidationSelector.getCrossValidator(),
            unsplit_x=unsplit_x,
            unsplit_y=unsplit_true_y,
        ).run()
        pipeline_report = ModelTester(
            pipeline_report=pipeline_report,
        ).run()
        ModelSummary(pipeline_report).run()
