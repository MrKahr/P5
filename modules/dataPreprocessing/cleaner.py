import re
from typing import Self
import pandas as pd

from modules.config.config import Config
from modules.logging import logger


class DataCleaner(object):
    _instance = None

    def __new__(cls, df: pd.DataFrame) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self, df: pd.DataFrame) -> None:
        if not self._created:
            self.df = df
            self.initial_row_count = self.df.shape[0]
            self._created = True

    def _deleteNanCols(self) -> None:
        """Remove columns where all entries are missing"""
        current_col_count = self.df.shape[1]
        self.df.dropna(axis=1, how="all", inplace=True)
        amount = current_col_count - self.df.shape[1]
        logger.info(f"Removed {amount} NaN {"columns" if amount != 1 else "column"}")

    def deleteNonfeatures(self) -> None:
        """
        Removes pig ID and sår ID from dataset because we consider neither a feature.
        """
        self.df.drop(["Gris ID", "Sår ID"], axis=1, inplace=True)

    def deleteMissingValues(self) -> None:
        """Drop all rows that contains value `100`: Manglende Værdi."""
        current_row_count = self.df.shape[0]
        for label in self.df.columns.values:
            if label:
                self.df.drop(
                    self.df[(self.df[label] == 100)].index,
                    inplace=True,
                )
        logger.info(
            f"Removed {current_row_count - self.df.shape[0]} rows containing '100' (Manglende værdi)"
        )

    def deleteUndeterminedValue(self) -> None:
        """Drop all rows that contains value `2`: Kan ikke vurderes"""
        current_row_count = self.df.shape[0]
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
        logger.info(
            f"Removed {current_row_count - self.df.shape[0]} rows containing '2' (Kan ikke vurderes)"
        )

    def showRowRemovalRatio(self) -> None:
        """
        Displays row removal ratio from start to present state of dataframe
        """
        percentage_row_removal = (1 - (self.df.shape[0] / self.initial_row_count)) * 100
        logger.info(
            f"Row removal ratio is currently {self.df.shape[0]}/{self.initial_row_count} ({percentage_row_removal:.2f}% removed)"
        )

    def removeFeaturelessRows(self, threshold: int = 4) -> None:
        """Removes rows containing a critical number of NaN
        This is meant to remove dead pigs from the dataset whose rows only contain grisid and sårid

        Parameters
        ----------
        threshold : int, optional
            The critical count of NaN in a row before it is removed, by default `4`
        """
        current_row_count = self.df.shape[0]
        self.df.dropna(axis=0, thresh=threshold, inplace=True)
        logger.info(f"Removed {current_row_count - self.df.shape[0]} rows")

    def convertHourToDay(self) -> None:
        """Cleans cells in a dataset containing hours < one day"""
        # Find all indeces of rows containing "time"
        indexes = []
        for i, value in self.df["Dag"].items():
            match_obj = re.search("time", value)
            if match_obj:
                indexes.append(i)
        if indexes:
            # Change time to 0 for rows with hours
            self.df.loc[indexes, "Dag"] = 0
        # The orignal "Tid" column was all strings. Convert them to integers
        self.df["Dag"] = pd.to_numeric(self.df["Dag"])
        logger.info(f"Converted {len(indexes)} rows from hour to day")

    def fillNan(self, fill_value: int = 100) -> None:
        """Fills all nan values in the dataset with an arbitrary fill value

        Parameters
        ----------
        fill_value : int, optional
            values to replace empty cells in the dataset, by default 100
        """
        current_row_count = self.df.shape[0]
        self.df.fillna(fill_value, inplace=True)
        logger.info(
            f"Filled {current_row_count - self.df.shape[0]} rows with '{fill_value}'"
        )

    def showNan(self) -> None:
        """Subsets and shows the current dataframe to include only"""
        nan_df = self.df[self.df.isna().any(axis=1)]
        if len(nan_df) == 0:
            logger.info("No NaN values to display.")
        else:
            logger.info(f"NaN values are \n{nan_df}")

    def cleanRegsDataset(self, fillna: int = 100) -> None:
        """Cleans the eksperiementelle_sår_2024 dataset according to hardcoded presets.

        Parameters
        ----------
        fillna : int, optional
            The value to fill in NA-cells, by default 100
        """
        self._deleteNanCols()
        self.removeFeaturelessRows()
        self.fillNan(fillna)

    def cleanMålDataset(self) -> None:
        """Cleans the eksperimentelle_sår_2024_mål dataset according to hardcoded presets"""
        self._deleteNanCols()
        current_row_count = self.df.shape[0]
        # Remove data not used in training
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
        logger.info(f"Removed {current_row_count - self.df.shape[0]} rows")

    def cleanOldDataset(self):
        """Cleans the old_eksperiementelle_sår_2014 dataset according to hardcoded presets"""
        self._deleteNanCols()
        self.convertHourToDay()

    def getDataframe(self) -> pd.DataFrame:
        """Get the cleaned dataframe as a deep copy.
        Returns
        -------
        pd.DataFrame
            The cleaned dataframe
        """
        self.showRowRemovalRatio()
        return self.df.copy(deep=True)

    def run(self) -> None:
        config = Config()
        if config.getValue("DeleteNanColumns"):
            self._deleteNanCols()
        if config.getValue("DeleteNonfeatures"):
            self.deleteNonfeatures()
        if config.getValue("DeleteMissingValues"):
            self.deleteMissingValues()
        if config.getValue("DeleteUndeterminedValue"):
            self.deleteUndeterminedValue()
        if config.getValue("RemoveFeaturelessRows"):
            self.removeFeaturelessRows(config.getValue("RFlRParams"))
        if config.getValue("FillNan"):
            self.fillNan()
        if config.getValue("ShowNan"):
            self.showNan()
        # TODO - Find out why row removal ration is n/n - some rows ought to be removed
        self.showRowRemovalRatio()
        logger.info(f"DataCleaner is done")
