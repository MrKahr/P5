import pandas as pd
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
from modules.modelSelection.model_selection import ModelSelector
from modules.modelTesting.score_functions import ScoreFunctions
from modules.modelTraining.model_training import ModelTrainer
from modules.modelTesting.model_testing import ModelTester
from modules.modelSummary.model_summarizing import ModelSummary


class Pipeline:
    """Responsible for running the entire pipeline, collecting datasets for each iteration and providing args to pipeline parts
    # NOTE - Be very careful when making changes here"""

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = Config()
        self.selected_features = None  # type: NDArray

    def getTrainingData(self) -> pd.DataFrame:
        return self.df.drop(["Dag"], axis=1, inplace=False)

    def getTargetData(self) -> pd.Series:
        return self.df["Dag"]

    def getTestData(self) -> pd.DataFrame:
        return self.df[self.selected_features]

    def run(self) -> None:
        """Using a dataframe, calls relevant pipeline components to perform transformations of dataset as specified in the config file"""
        self.df = DataCleaner(self.train_dataset).run()
        self.df = OutlierProcessor(self.df).run()
        self.df = DataTransformer(self.df).run()

        train_x, train_true_y, self.selected_features = FeatureSelector(
            self.getTrainingData(), self.getTargetData()
        ).run()
        fit_estimator, model_report = ModelTrainer(
            estimator=ModelSelector.getModel(),
            cv=CrossValidationSelector.getCrossValidator(),
            train_x=train_x,
            true_y=train_true_y,
        ).run()

        self.df = DataCleaner(self.test_dataset).run()
        model_report = ModelTester(
            estimator=fit_estimator,
            train_x=train_x,
            train_true_y=train_true_y,
            test_x=self.getTestData(),
            test_true_y=self.getTargetData(),
            model_report=model_report,
        ).run()
        ModelSummary(model_report).run()
