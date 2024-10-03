# Libraries
from enum import Enum
import os
from pathlib import Path
import re
import pandas as pd  # CSV-reading, data manipulation and cleaning.
from numpy.typing import NDArray


class Dataset(Enum):
    MÅL = "eksperimentelle_sår_2024_mål.csv"
    REGS = "eksperimentelle_sår_2024_regs.csv"
    OLD = "old_eksperimentelle_sår_2014_regs.csv"


class DataProcessor:
    def __init__(
        self,
        type: Dataset,
        remove_undetermined: bool = False,
        remove_missing: bool = False,
    ) -> None:
        path = Path(
            Path(os.path.split(__file__)[0]).parents[0],
            f"data/{type.value}",
        )
        self.df = pd.read_csv(path, sep=";", comment="#")
        self.remove_missing = remove_missing
        self.remove_undetermined = remove_undetermined
        self.dataset_type = type
        self.deleteNaN()
        if type == Dataset.OLD:
            self.time_label = "Tid"
            self._deleteFromOld()
        else:
            self.time_label = "Dag"

    def _formatTrainingData(self) -> pd.DataFrame:
        return self.df.drop(
            ["Gris ID", "Sår ID", self.time_label], axis=1, inplace=False
        )

    def _deleteFromOld(self) -> None:
        indexes = []
        for i, value in self.df[self.time_label].items():
            match_obj = re.search("time", value)
            if match_obj:
                indexes.append(i)
        if indexes:
            # Drop all rows where "Tid" cell is less than 1 day
            self.df.drop(axis=0, index=indexes, inplace=True)
        # The orignal "Tid" column was all strings. Convert them to integers
        self.df[self.time_label] = pd.to_numeric(self.df[self.time_label])

    def showDataFrame(self) -> None:
        print(self.df)

    def getDataFrame(self) -> pd.DataFrame:
        return self.df

    def showNaN(self) -> None:
        nan_df = self.df[self.df.isna().any(axis=1)]
        if len(nan_df) == 0:
            print("Empty dataframe (no NaN values to display)")
        else:
            print(nan_df)

    def deleteNaN(self) -> None:
        # Remove columns where all entries are missing
        self.df.dropna(axis=1, how="all", inplace=True)

        if self.dataset_type == Dataset.REGS:
            # Drop rows for pigs with at least 4 entries are missing (i.e. the dead pigs)
            self.df.dropna(axis=0, thresh=4, inplace=True)

            # Replace all missing single values with 100 (indicating a missing value)
            self.df["Infektionsniveau"] = (
                self.df["Infektionsniveau"].fillna(100, axis=0).values
            )
        elif self.dataset_type == Dataset.MÅL:
            # Remove unnecessary data
            self.df.drop(
                columns=["Længde (cm)", "Bredde (cm)", "Dybde (cm)", "Areal (cm^2)"],
                inplace=True,
            )
            # Remove any NaN value in granulation tissue data
            self.df.dropna(
                axis=0, how="any", subset=["Sårrand (cm)", "Midte (cm)"], inplace=True
            )
            # Insert missing IDs for pigs using the single existing ID
            self.df["Gris ID"] = self.df["Gris ID"].ffill(axis=0).values

    def deleteMissing(self) -> None:
        """Only targets REGS
        NOTE: This prunes ~80 entries in the dataset
        """
        labels = [
            "Kontraktion",
            "Ødem",
            "Epithelialisering",
            "Eksudat",
            "Granulationsvæv",
        ]
        if self.dataset_type == Dataset.REGS:
            for label in labels:
                self.df.drop(
                    self.df[(self.df[label] == 100)].index,
                    inplace=True,
                )

    def deleteUndetermined(self) -> None:
        """Only targets REGS
        NOTE: This prunes ~50% of the dataset
        """
        labels = [
            "Kontraktion",
            "Ødem",
            "Epithelialisering",
            "Eksudat",
            "Granulationsvæv",
        ]
        if self.dataset_type == Dataset.REGS:
            for label in labels:
                self.df.drop(
                    self.df[(self.df[label] == 2)].index,
                    inplace=True,
                )
                
    def oneHotEncoding(self, variableNames: list ) -> None:
        """One-hot encode one or more categorical attributes

        Based on https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
        """
        for variable in variableNames:
            one_hot = pd.get_dummies(self.df[variable], prefix=variable)
            self.df.drop(variable, inplace=True, axis = 1)
            self.df = self.df.join(one_hot)

    def getTrainingData(self) -> NDArray:
        return self._formatTrainingData().to_numpy()

    def getTargetData(self) -> NDArray:
        return self.df[self.time_label].to_numpy(copy=True)

    def getTrainingLabels(self) -> list[str]:
        return self._formatTrainingData().columns.values

    def getTargetMaxValue(self) -> int:
        ndarr = self.df[self.time_label].unique()
        i = ndarr.argmax()
        return ndarr[i]

