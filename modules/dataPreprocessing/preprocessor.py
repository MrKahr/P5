import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
from modules.dataPreprocessing.dataset_enums import Dataset


class DataPreprocessor:
    def __init__(self, data: Dataset | pd.DataFrame) -> None:
        # Ensure dataset is loaded correctly
        if not isinstance(data, pd.DataFrame):
            path = Path("data", data.value).absolute()
            self.df = pd.read_csv(path, sep=";", comment="#")
        else:
            self.df = data

    def _formatTrainingData(self) -> pd.DataFrame:
        return self.df.drop(["Gris ID", "Sår ID", "Dag"], axis=1, inplace=False)

    def showDataFrame(self) -> None:
        print(self.df)

    def getDataFrame(self) -> pd.DataFrame:
        return self.df

    def getTrainingData(self) -> NDArray:
        return self._formatTrainingData().to_numpy()

    def getTargetData(self) -> NDArray:
        return self.df["Dag"].to_numpy(copy=True)

    def getTrainingLabels(self) -> list[str]:
        return self._formatTrainingData().columns.values

    def getTargetMaxValue(self) -> int:
        ndarr = self.df["Dag"].unique()
        i = ndarr.argmax()
        return ndarr[i]
