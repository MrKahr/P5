from pathlib import Path
import re
import pandas as pd

from modules.config.config import Config
from modules.dataPreprocessing.dataset_enums import Dataset
from modules.logging import logger


class DataCleaner(object):

    def __init__(self, dataset: Dataset) -> None:
        logger.info(f"Loading '{dataset.name}' dataset")
        path = Path("data", dataset.value).absolute()
        self.dataset = dataset
        self.df = pd.read_csv(path, sep=";", comment="#")
        self.initial_row_count = self.df.shape[0]

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
        non_features = ["Gris ID", "Sår ID"]
        self.df.drop(non_features, axis=1, inplace=True)
        logger.info(
            f"Removed {len(non_features)} non-informative features: {", ".join(non_features)}"
        )

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
        logger.info(
            f"Removed {current_row_count - self.df.shape[0]} rows containing {threshold} or more NaN values"
        )

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
            logger.info("No NaN values to display")
        else:
            logger.info(f"NaN values are \n{nan_df}")

    def cleanMålDataset(self) -> None:
        """Cleans the eksperimentelle_sår_2024_mål dataset according to hardcoded presets"""
        current_row_count = self.df.shape[0]
        # Remove data not used in training
        cols = ["Længde (cm)", "Bredde (cm)", "Dybde (cm)", "Areal (cm^2)"]
        self.df.drop(
            columns=cols,
            inplace=True,
        )
        logger.info(
            f"Removed {len(cols)} features not used for training: {", ".join(cols)}"
        )

        # Remove any NaN value in granulation tissue data
        subset = ["Sårrand (cm)", "Midte (cm)"]
        df = self.df.dropna(axis=0, how="any", subset=subset, inplace=False)
        dropped_rows = len(self.df.isna()) - len(df.isna())
        self.df = df
        logger.info(
            f"Removed {dropped_rows} NaN rows from features {", ".join(subset)}"
        )

        # Insert missing IDs for pigs using the single existing ID
        self.df["Gris ID"] = self.df["Gris ID"].ffill(axis=0).values
        logger.info(f"Removed {current_row_count - self.df.shape[0]} rows")

    def cleanOldDataset(self):
        """Cleans the old_eksperiementelle_sår_2014 dataset according to hardcoded presets"""
        self.convertHourToDay()

    def run(self) -> pd.DataFrame:
        """Run all applicable data cleaning methods

        Returns
        -------
        pd.DataFrame
            The cleaned dataset that is returned to the pipeline
        """
        config = Config()
        if config.getValue("UseCleaner"):
            if config.getValue("DeleteNanColumns"):
                self._deleteNanCols()
            if config.getValue("DeleteNonfeatures"):
                self.deleteNonfeatures()
            if config.getValue("DeleteUndeterminedValue"):
                self.deleteUndeterminedValue()
            if config.getValue("RemoveFeaturelessRows"):
                self.removeFeaturelessRows(config.getValue("RemoveFeaturelessRowsArgs"))
            if config.getValue("FillNan"):
                self.fillNan()
            if config.getValue("DeleteMissingValues"):
                self.deleteMissingValues()
            if config.getValue("ShowNan"):
                self.showNan()

            if self.dataset == Dataset.MÅL:
                self.cleanMålDataset()
            elif self.dataset == Dataset.OLD:
                self.cleanOldDataset()

            self.showRowRemovalRatio()
        else:
            logger.info("Skipping data cleaning")

        return self.df
