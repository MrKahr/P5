import re
import pandas as pd

from modules.logging import logger


class DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.initial_row_count = self.df.shape[0]
        self.current_row_count = self.df.shape[0]  # We need stepwise row removal counts

    def _deleteNanCols(self) -> None:
        """Remove columns where all entries are missing"""
        self.current_row_count = self.df.shape[0]
        self.df.dropna(axis=1, how="all", inplace=True)
        logger.info(
            f"{__name__} removed {self.current_row_count - self.df.shape[0]} rows"
        )

    def _deleteNonfeatures(self) -> pd.DataFrame:
        """
        Removes pig ID and sår ID from dataset because we consider neither a feature.

        Return
        -------
        pd.DataFrame
            Dataframe with two columns removed
        """
        return self.df.drop(["Gris ID", "Sår ID"], axis=1, inplace=False)

    def _deleteMissingValue(self) -> None:
        """Drop all rows that contains value `100`: Manglende Værdi.
        NOTE: This prunes ~80 entries in the dataset.
        """
        self.current_row_count = self.df.shape[0]
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
        logger.info(
            f"{__name__} removed {self.current_row_count - self.df.shape[0]} rows"
        )

    def _deleteUndeterminedValue(self) -> None:
        """Drop all rows that contains value `2`: Kan ikke vurderes
        NOTE: This prunes ~50% of the dataset
        """
        self.current_row_count = self.df.shape[0]
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
            f"{__name__} removed {self.current_row_count - self.df.shape[0]} rows"
        )

    def showRowRemovalRatio(self) -> None:
        """
        Displays row removal ratio from starting start to present state of dataframe
        """
        percentage_row_removal = (1 - (self.df.shape[0] / self.initial_row_count)) * 100
        logger.info(
            f"Row removal ratio is currently {self.df.shape[0]}/{self.initial_row_count} ({percentage_row_removal:.2f}% removed)"
        )

    def removeFeaturelessRows(self, threshold: int = 4, fillna: int = 100) -> None:
        """Removes rows containing a critical number of NaN
        #NOTE - This is meant to remove dead pigs from the dataset whose rows only contain grisid and sårid

        Parameters
        ----------
        threshold : int, optional
            the critical count of nans in a row before it is removed, by default 4
        fillna : int, optional
            the value to fill in NA-cells, by default 100
        """
        self.current_row_count = self.df.shape[0]
        self._deleteNanCols()
        # Drop rows for pigs with at least 4 entries are missing (i.e. the dead pigs)
        self.df.dropna(axis=0, thresh=threshold, inplace=True)

        self.df["Infektionsniveau"] = (
            self.df["Infektionsniveau"].fillna(fillna, axis=0).values
        )
        logger.info(
            f"{__name__} removed {self.current_row_count - self.df.shape[0]} rows"
        )

    # TODO - Check whether the current dataset is indeed old, otherwise do nohting
    def cleanOldDataset(self):
        """Cleans the old_eksperiementelle_sår_2014 dataset according to hardcoded presets"""
        self.current_row_count = self.df.shape[0]
        self._deleteNanCols()
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
        logger.info(
            f"{__name__} removed {self.current_row_count - self.df.shape[0]} rows"
        )

    # TODO: check whether the current dataset is mål, otherwise do nothing
    def cleanMålDataset(self) -> None:
        """Cleans the eksperimentelle_sår_2024_mål dataset according to hardcoded presets"""
        self.current_row_count = self.df.shape[0]
        self._deleteNanCols()
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
        logger.info(
            f"{__name__} removed {self.current_row_count - self.df.shape[0]} rows"
        )

    def fillNan(self, fill_value: int = 100) -> None:
        """Fills all nan values in the dataset with an arbitrary fill value

        Parameters
        ----------
        fill_value : int, optional
            values to replace empty cells in the dataset, by default 100
        """
        # Replace all missing single values with 100 (indicating a missing value)
        self.df.fillna(fill_value)

    def showNan(self) -> None:
        """Subsets and shows the current dataframe to include only"""
        nan_df = self.df[self.df.isna().any(axis=1)]
        if len(nan_df) == 0:
            print("Empty dataframe (no NaN values to display)")
        else:
            print(nan_df)

    def getDataframe(self) -> pd.DataFrame:
        """Get the transformed dataframe as a deep copy.
        Returns
        -------
        pd.DataFrame
            The transformed dataframe
        """
        return self.df.copy(deep=True)
