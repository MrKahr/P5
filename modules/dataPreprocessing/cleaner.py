import re
import pandas as pd

from modules.config.pipeline_config import PipelineConfig
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.logging import logger


class DataCleaner(object):

    def __init__(self, df: pd.DataFrame, dataset: Dataset) -> None:
        """
        Performs low-level cleaning of `df` such as removing NaN values.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to clean.

        dataset : Dataset
            The dataset from which `df` originates.
            Used for required cleaning of `MÅL` and `OLD` datasets.
        """
        self.df = df
        self.dataset = dataset
        self.initial_row_count = self.df.shape[0]  # Used to keep track of rows removed
        logger.info(f"Cleaning dataset '{self.dataset.name}'...")

    def _deleteNanCols(self) -> None:
        """Remove columns where all entries are NaN."""
        current_col_count = self.df.shape[1]  # Get number of columns
        self.df.dropna(axis=1, how="all", inplace=True)
        amount = current_col_count - self.df.shape[1]  # Compute column removal count
        logger.info(f"Removed {amount} NaN {"columns" if amount != 1 else "column"}")

    def deleteNonfeatures(self) -> None:
        """Remove columns `Gris ID` and `Sår ID` because we consider neither feature."""
        non_features = ["Gris ID", "Sår ID"]
        self.df.drop(non_features, axis=1, inplace=True)
        logger.info(
            f"Removed {len(non_features)} non-informative features: {non_features}"
        )

    def deleteMissingValues(self) -> None:
        """Drop all rows that contains value `100` (a.k.a. `Manglende værdi`)."""
        current_row_count = self.df.shape[0]  # Get number of rows

        # Check all labels for missing values and remove all rows containing any
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
        """Drop all rows that contains value `2` (a.k.a. `Kan ikke vurderes`)"""
        current_row_count = self.df.shape[0]  # Get number of rows

        # We check these labels for `2`
        labels = [
            "Kontraktion",
            "Ødem",
            "Epithelialisering",
            "Eksudat",
            "Granulationsvæv",
        ]

        # For each label, drop all rows that contain `2`
        for label in labels:
            self.df.drop(
                self.df[(self.df[label] == 2)].index,
                inplace=True,
            )
        logger.info(
            f"Removed {current_row_count - self.df.shape[0]} rows containing '2' (Kan ikke vurderes)"
        )

    def showRowRemovalRatio(self) -> None:
        """Displays row removal ratio from start to present state of dataframe"""
        percentage_row_removal = (1 - (self.df.shape[0] / self.initial_row_count)) * 100
        logger.info(
            f"Row removal ratio for dataset '{self.dataset.name}' is currently {self.df.shape[0]}/{self.initial_row_count} ({percentage_row_removal:.2f}% removed)"
        )

    def removeFeaturelessRows(self, threshold: int) -> None:
        """
        Removes rows containing a critical number of NaN.

        This is meant to remove dead pigs from the dataset
        (where every column except `Gris ID`, `Sår ID`, and `Dag` are NaN).

        Parameters
        ----------
        threshold : int
            The critical count of NaN in a row before it is removed.
        """
        current_row_count = self.df.shape[0]  # Get number of rows
        self.df.dropna(axis=0, thresh=threshold, inplace=True)
        logger.info(
            f"Removed {current_row_count - self.df.shape[0]} rows containing {threshold} or more NaN values"
        )

    def convertHourToDay(self) -> None:
        """
        If a value for label `Dag` is measured in hours, convert it to a measurement in days.
        NOTE: It is currently hardcoded to 0, i.e., any value measured in hours is converted to `0` days.
        """
        # Store the index for each row where the `Dag` value is measured in hours
        indices = []

        # Find all indices of rows where the `Dag` value contains the danish word "time" (i.e. "hour")
        for i, value in self.df["Dag"].items():
            if re.search("time", value):
                indices.append(i)

        if indices:
            # Change time to 0 for rows with hours
            self.df.loc[indices, "Dag"] = 0

        # The type of the rows measured in hours is a string. Convert them to integers
        self.df["Dag"] = pd.to_numeric(self.df["Dag"])

        logger.info(f"Converted {len(indices)} rows from hour to day")

    def fillNan(self, fill_value: int = 100) -> None:
        """
        Fills all NaN values in the dataset with an arbitrary fill value.

        Parameters
        ----------
        fill_value : int, optional
            The value to replace empty cells in the dataset, by default 100.
        """
        current_row_count = self.df.shape[0]  # Get number of rows
        self.df.fillna(fill_value, inplace=True)
        logger.info(
            f"Filled {current_row_count - self.df.shape[0]} rows with '{fill_value}'"
        )

    def showNan(self) -> None:
        """Check the dataset for NaN values and show all NaN values detected."""
        nan_df = self.df[self.df.isna().any(axis=1)]
        if len(nan_df) == 0:
            logger.info("No NaN values to display")
        else:
            logger.info(f"NaN values are:\n{nan_df}")

    def cleanMålDataset(self) -> None:
        """Cleans the eksperimentelle_sår_2024_mål dataset according to hardcoded presets"""
        current_row_count = self.df.shape[0]  # Get number of rows

        # Remove columns not used in training
        cols = ["Længde (cm)", "Bredde (cm)", "Dybde (cm)", "Areal (cm^2)"]
        self.df.drop(
            columns=cols,
            inplace=True,
        )
        logger.info(f"Removed {len(cols)} features not used during training: {cols}")

        # Remove all rows with any NaN value in both granulation tissue columns
        subset = ["Sårrand (cm)", "Midte (cm)"]
        df = self.df.dropna(axis=0, how="any", subset=subset, inplace=False)
        dropped_rows = len(self.df.isna()) - len(df.isna())
        logger.info(f"Removed {dropped_rows} NaN rows from features {subset}")
        self.df = df

        # Insert missing `Gris ID` for pigs using the single existing `Gris ID`
        self.df["Gris ID"] = self.df["Gris ID"].ffill(axis=0).values
        logger.info(
            f"Added {len(self.df["Gris ID"].isna())} missing values for column 'Gris ID'"
        )

    def cleanOldDataset(self):
        """Cleans the old_eksperiementelle_sår_2014 dataset according to hardcoded presets"""
        self.convertHourToDay()

    def run(self) -> pd.DataFrame:
        """
        Run all applicable data cleaning methods.

        Returns
        -------
        pd.DataFrame
            The cleaned dataset.
        """
        config = PipelineConfig()
        if config.getValue("UseCleaner"):
            # General cleaning
            if config.getValue("DeleteNanColumns"):
                self._deleteNanCols()
            if config.getValue("RemoveFeaturelessRows"):
                self.removeFeaturelessRows(config.getValue("RemoveFeaturelessRowsArgs"))

            # Dataset-specific cleaning
            if self.dataset == Dataset.REGS:
                if config.getValue("DeleteNonfeatures"):
                    self.deleteNonfeatures()
                if config.getValue("DeleteUndeterminedValue"):
                    self.deleteUndeterminedValue()
                if config.getValue("FillNan"):
                    self.fillNan()
                if config.getValue("DeleteMissingValues"):
                    self.deleteMissingValues()
            elif self.dataset == Dataset.MÅL:
                self.cleanMålDataset()
            elif self.dataset == Dataset.OLD:
                self.cleanOldDataset()

            # Cleaning results
            if config.getValue("ShowNan"):
                self.showNan()
            self.showRowRemovalRatio()
        else:
            logger.info("Skipping data cleaning")

        return self.df
