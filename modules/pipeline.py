from typing import Any
import pandas as pd
from pathlib import Path
from sklearn import model_selection
import torch
from absl import logging as absl_logging

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import LogLevel
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
    _config = PipelineConfig()

    def __init__(self, train_dataset: Dataset) -> None:
        """
        Responsible for running the entire pipeline, collecting datasets for each iteration
        and providing args to pipeline parts.

        Parameters
        ----------
        train_dataset : Dataset
            Dataset used for model training.
        """
        self.train_dataset = train_dataset
        self.df = None  # type: pd.DataFrame
        self._data_folder = "dataset"

    def _setLogLevel(self) -> None:
        level = LogLevel._member_map_.get(
            self._config.getValue("loglevel", "General"), None
        )
        if level is None:
            level = LogLevel.DEBUG
            self._logger.warning(
                f"Invalid loglevel '{level}', defaulting to '{level.name}'"
            )

        self._logger.setLevel(level)
        torch._logging.set_logs(all=level)
        absl_logging.set_verbosity(level)

    def train_test_split(self, unsplit_x: pd.DataFrame, unsplit_y: pd.Series) -> list:
        """
        Create train/test split.

        Parameters
        ----------
        unsplit_x : pd.DataFrame
            Features of full dataset.

        unsplit_y : pd.Series
            Target label of full dataset.

        Returns
        -------
        list
            A list of 4 elements.
            [0] : train_x
            [1] : test_x
            [2] : train_y
            [3] : test_y
        """
        # *_x == pd.DataFrame, *_y == pd.Series
        train_test_args = self._config.getValue("train_test_split", "ModelTraining")
        train_test_ratio = (
            f"{train_test_args["train_size"]:.2f}/{1-train_test_args["train_size"]:.2f}"
        )
        self._logger.info(f"Train/test split: {train_test_ratio}")
        return model_selection.train_test_split(
            unsplit_x,
            unsplit_y,
            shuffle=True,
            stratify=unsplit_y,
            **train_test_args,
        )

    def loadDataset(self, dataset: Dataset) -> pd.DataFrame:
        logger.info(f"Loading '{dataset.name}' dataset")
        path = Path(self._data_folder, dataset.value).absolute()
        return pd.read_csv(path, sep=";", comment="#")

    def addMål(self) -> pd.DataFrame:
        """
        Join MÅL with the current dataset using "Gris ID", "Sår ID", and "Dag" as index.
        """
        mål_df = DataCleaner(self.loadDataset(Dataset.MÅL), Dataset.MÅL).run()

        join_columns = ["Gris ID", "Sår ID", "Dag"]
        logger.info(
            f"Joining '{Dataset.MÅL.name}' with training dataset '{self.train_dataset.name}' on columns {join_columns}"
        )
        # set a multi-index on Mål
        mål_df.set_index(join_columns, inplace=True)

        # with the multi-index on Mål, we can join on multiple things at once
        return self.df.join(mål_df, how="left", on=join_columns)

    def getX(self) -> pd.DataFrame:
        return self.df.drop(["Dag"], axis=1, inplace=False)

    def getY(self) -> pd.Series:
        return self.df["Dag"]

    def run(self) -> dict[str, Any]:
        """
        Using a dataframe, calls relevant pipeline components to perform
        transformations of dataset as specified in the config file.
        """
        self._logger.info(
            f"Initializing pipeline. Training dataset: '{self.train_dataset.name}'"
        )
        self.df = self.loadDataset(self.train_dataset)

        # Join dataset with MÅL if we want to use that
        if PipelineConfig().getValue("UseContinuousFeatures"):
            self.df = self.addMål()

        self.df = DataCleaner(
            self.df,
            self.train_dataset,
        ).run()
        pipeline_report = DataTransformer(
            *self.train_test_split(self.getX(), self.getY())
        ).run()
        pipeline_report = OutlierProcessor(pipeline_report).run()
        pipeline_report = FeatureSelector(pipeline_report).run()
        pipeline_report = ModelTrainer(
            estimator=ModelSelector.getModel(),
            cross_validator=CrossValidationSelector.getCrossValidator(),
            pipeline_report=pipeline_report,
        ).run()
        pipeline_report = ModelTester(pipeline_report).run()
        ModelSummary(pipeline_report).run()
        return pipeline_report
