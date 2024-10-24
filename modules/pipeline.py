import pandas as pd
from pathlib import Path


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
from modules.modelSelection.model_selection import ModelSelector
from modules.modelTesting.score_functions import ScoreFunctions
from modules.modelTraining.model_training import ModelTrainer
from modules.modelTesting.model_testing import ModelTester
from modules.modelSummary.model_summarizing import ModelSummary
from modules.logging import logger


class Pipeline:
    """Responsible for running the entire pipeline, collecting datasets for each iteration and providing args to pipeline parts
    # NOTE - Be very careful when making changes here"""

    def __init__(self, data: Dataset) -> None:
        logger.info(f"Loading '{data.name}' dataset")
        path = Path("data", data.value).absolute()
        self.df = pd.read_csv(path, sep=";", comment="#")
        self.config = Config()

    def getTrainingData(self) -> pd.DataFrame:
        return self.df.drop(["Dag"], axis=1, inplace=False)

    def getTargetData(self) -> pd.Series:
        return self.df["Dag"]

    def run(self) -> None:
        """Using a dataframe, calls relevant pipeline components to perform transformations of dataset as specified in the config file"""
        self.df = DataCleaner(self.df).run()
        self.df = FeatureSelector(self.getTrainingData(), self.getTargetData()).run()
        self.df = OutlierProcessor(self.df).run()
        self.df = DataTransformer(self.df).run()
        fit_estimator, model_report = ModelTrainer(
            estimator=ModelSelector.getModel(),
            cv=CrossValidationSelector.getCrossValidator(),
            score_funcs=ScoreFunctions.getModelScoreFuncs(),
            train_x=self.getTrainingData(),
            target_y=self.getTargetData(),
        ).run()
        ModelTester(
            estimator=fit_estimator,
            train_x=self.getTrainingData(),
            target_y=self.getTargetData(),
            model_report=model_report,
        ).run()
        # ModelSummary().run()
