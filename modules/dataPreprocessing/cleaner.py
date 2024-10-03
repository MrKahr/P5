import re
import pandas as pd
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.dataPreprocessing.preprocessor import DataPreprocessor


class DataCleaner(DataPreprocessor):
    def __init__(self, data: Dataset | pd.DataFrame) -> None:
        super().__init__(data)

    def _deleteNaN(self) -> None:
        """Remove columns where all entries are missing"""
        self.df.dropna(axis=1, how="all", inplace=True)

    def _deleteMissing(self) -> None:
        """Drop all rows that contains value `100` in dataset `REGS`.
        NOTE: This prunes ~80 entries in the dataset.
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

    def _deleteUndetermined(self) -> None:
        """Drop all rows that contains value `2` in dataset `REGS`.
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

    def cleanREGS(self, threshold: int = 4, fillna: int = 100) -> None:
        self._deleteNaN()
        # Drop rows for pigs with at least 4 entries are missing (i.e. the dead pigs)
        self.df.dropna(axis=0, thresh=threshold, inplace=True)

        self.df["Infektionsniveau"] = (
            self.df["Infektionsniveau"].fillna(fillna, axis=0).values
        )

    def cleanOLD(self):
        self._deleteNaN()
        # Find all indeces of rows containing "time"
        indexes = []
        for i, value in self.df[self.time_label].items():
            match_obj = re.search("time", value)
            if match_obj:
                indexes.append(i)
        if indexes:
            # Change time to 0 for rows with hours
            self.df["Dag"][indexes] = 0  # TODO: Test this, by printing the old dataset
            # self.df.drop(axis=0, index=indexes, inplace=True) # Drop all rows where "Tid" cell is less than 1 day
        # The orignal "Tid" column was all strings. Convert them to integers
        self.df[self.time_label] = pd.to_numeric(self.df[self.time_label])

    def cleanMÅL(self) -> None:
        self._deleteNaN()
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

    def fillna(self, fill_value: int = 100) -> None:
        # Replace all missing single values with 100 (indicating a missing value)
        self.df.fillna(fill_value)

    def showNaN(self) -> None:
        nan_df = self.df[self.df.isna().any(axis=1)]
        if len(nan_df) == 0:
            print("Empty dataframe (no NaN values to display)")
        else:
            print(nan_df)
