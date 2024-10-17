import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.logging import logger


class DataPreprocessor:
    """This is the main entry point for data processing
    This class contains the main dataframe.
    The other preprocessor classes work akin to a function, taking an input and returning an output
    """

    def __init__(self, data: Dataset) -> None:
        logger.info(f"Loading '{data.name}' dataset")
        path = Path("data", data.value).absolute()
        self.df = pd.read_csv(path, sep=";", comment="#")

    def _formatTrainingData(self) -> pd.DataFrame:
        return self.df.drop(["Dag"], axis=1, inplace=False)

    # TODO: Example method (follow-up with other models)
    def preprocessForNaiveBayes(self) -> pd.DataFrame: ...

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
