import re
import pandas as pd
from modules.dataPreprocessing.dataset_enums import Dataset


class DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.initial_row_count = self.df.shape[0]

    def _deleteNan(self) -> None:
        """Remove columns where all entries are missing"""
        self.df.dropna(axis=1, how="all", inplace=True)

    def _deleteNonfeatures(self) -> pd.DataFrame:
        """
        Removes pig ID and sår ID from dataset because we consider neither a feature.

        Return
        -------
        pd.DataFrame
            Dataframe with two columns removed
        """
        return self.df.drop(["Gris ID", "Sår ID"], axis=1, inplace=False)

    def _deleteMissing(self) -> None:
        """Drop all rows that contains value `100`.
        NOTE: This prunes ~80 entries in the dataset.
        """
        labels = [
            "Kontraktion",
            "Ødem",
            "Epithelialisering",
            "Eksudat",
            "Granulationsvæv",
        ]
        for label in labels:
            self.df.drop(
                self.df[(self.df[label] == 100)].index,
                inplace=True,
            )

    def _deleteUndetermined(self) -> None:
        """Drop all rows that contains value `2`.
        NOTE: This prunes ~50% of the dataset
        """
        labels = [
            "Kontraktion",
            "Ødem",
            "Epithelialisering",
            "Eksudat",
            "Granulationsvæv",
        ]
        for label in labels:
            self.df.drop(
                self.df[(self.df[label] == 2)].index,
                inplace=True,
            )

    def _showCurrentRowCount(self) -> None:
        """
        Displays row removal ratio from starting start to present state of dataframe
        """
        percentage_row_removal = (1 - (self.df.shape[0] / self.initial_row_count)) * 100
        print(
            f"Row removal ratio is currently {self.df.shape[0]}/{self.initial_row_count} ({percentage_row_removal:.2f}% removed)"
        )

    def cleanRegs(self, threshold: int = 4, fillna: int = 100) -> None:
        self._deleteNan()
        # Drop rows for pigs with at least 4 entries are missing (i.e. the dead pigs)
        self.df.dropna(axis=0, thresh=threshold, inplace=True)

        self.df["Infektionsniveau"] = (
            self.df["Infektionsniveau"].fillna(fillna, axis=0).values
        )

    def cleanOld(self):
        self._deleteNan()
        # Find all indeces of rows containing "time"
        indexes = []
        for i, value in self.df["Dag"].items():
            match_obj = re.search("time", value)
            if match_obj:
                indexes.append(i)
        if indexes:
            # Change time to 0 for rows with hours
            self.df.loc[indexes, "Dag"] = 0
            # self.df.drop(axis=0, index=indexes, inplace=True) # Drop all rows where "Tid" cell is less than 1 day
        # The orignal "Tid" column was all strings. Convert them to integers
        self.df["Dag"] = pd.to_numeric(self.df["Dag"])

    def cleanMål(self) -> None:
        self._deleteNan()
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

    def fillNan(self, fill_value: int = 100) -> None:
        # Replace all missing single values with 100 (indicating a missing value)
        self.df.fillna(fill_value)

    def showNan(self) -> None:
        nan_df = self.df[self.df.isna().any(axis=1)]
        if len(nan_df) == 0:
            print("Empty dataframe (no NaN values to display)")
        else:
            print(nan_df)
