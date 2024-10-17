import pandas as pd
from sklearn.impute import KNNImputer
from numpy.typing import ArrayLike

from modules.dataPreprocessing.processor import Processor
from modules.logging import logger


class DataTransformer(Processor):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def getDataframe(self) -> pd.DataFrame:
        """Get a deep copy of the transformed dataframe.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe
        """
        return self.df.copy(deep=True)

    def oneHotEncode(self, labels: list[str]) -> None:
        """One-hot encode one or more categorical attributes, selected by `lables`.

        Based on https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

        Parameters
        ----------
        labels : list[str]
            A list of the column labels to one-hot encode.
        """
        for variable in labels:
            one_hot = pd.get_dummies(self.df[variable], prefix=variable)
            self.df.drop(variable, inplace=True, axis=1)
            self.df = self.df.join(one_hot)

    def modeImputationByDay(self) -> None:
        """Imputes missing values by replacing them with the most common value for the day where the value is missing.
        Takes no arguments and modifies the Dataframe on the class itself.

        Returns
        -------
        Nothing
        """
        # for this method
        # get dataset
        # for every row
        # if there is a missing value, find its day
        # find the most common value in that column for that day
        # replace the missing value
        # repeat
        df = self.df
        for index, row in df.iterrows():
            for label in df.columns.values:
                if row[label] == 100:
                    day = row["Dag"]

                    logger.info(
                        f"Found missing {label} at Pig ID {row['Gris ID']}, Wound ID {row['Sår ID']}, Day {day} (Internal Index {index}). Imputing..."
                    )  # NOTE 'Internal Index' is the index of the row in the dataframe.
                    # It's almost the same as in the excel sheet, but not quite, because we remove unecessary or invalid rows.
                    same_day_rows = df[df["Dag"] == day]
                    column = same_day_rows[label]
                    mode = column.mode()[
                        0
                    ]  # NOTE mode() returns a dataframe, actually. Since we use it for a single column, there is only one value. Indexing the output with 0 gets us that value.

                    logger.info(f"Mode of {label} is {mode}.")
                    if mode == 100:
                        logger.warning(
                            "Mode is a missing value! Cannot properly impute!"
                        )

                    df.at[index, label] = mode

                    logger.info(f"Replaced missing value with {mode}.")
        self.df = df

    def zeroOneDistance(
        self, x: ArrayLike, y: ArrayLike, *args, missing_values=100
    ) -> int:
        """An implementation of a zero-one distance metric in a format scikit's KNNImputer can use

        Parameters
        ----------
        x : 1D Numpy Array
            A representation of a row in the dataset as an array of numbers
        y : 1D Numpy Array
            A representation of a row in the dataset as an array of numbers
        missing_values :
            What value should be considered missing and ineligible for comparison, by default 100

        Returns
        -------
        int
            The distance from x to y measured by counting the number of different entries in the two arrays.
        """
        distance = 0
        for index, entry in enumerate(x):  # NOTE enumerate makes the index available
            if entry != y[index] and not (
                entry == missing_values or y[index] == missing_values
            ):
                distance += 1
        return distance

    def matrixDistance(
        self, x: ArrayLike, y: ArrayLike, *args, missing_values=100
    ) -> int:
        """An implementation of a distance metric in a format scikit's KNNImputer can use. This one is uses distance matrices for each variable

        Parameters
        ----------
        x : 1D Numpy Array
            A representation of a row in the dataset as an array of numbers
        y : 1D Numpy Array
            A representation of a row in the dataset as an array of numbers
        missing_values :
            What value should be considered missing and ineligible for comparison, by default 100

        Returns
        -------
        int
            The distance from x to y measured by counting the number of different entries in the two arrays.
        """
        dag_matrix = [[]]
        niveau_sårvæv_matrix = [[]]
        sårskorpe_matrix = [[]]
        granulationsvæv_matrix = [[]]
        epithelialisering_matrix = [[]]
        kontraktion_matrix = [[]]
        hyperæmi_matrix = [[]]
        ødem_matrix = [[]]
        eksudat_matrix = [[]]
        eksudattype_matrix = [[]]
        infektionsniveau_matrix = [[]]

    def KNNImputation(self) -> None:
        """Imputes missing values using Scikit's KNNImputer. Takes no arguments and modifies the Dataframe on the class itself.

        Returns
        -------
        Nothing
        """
        df = self.df
        self.LogValues(df)
        logger.info("Starting KNN-Imputation.")
        imputer = KNNImputer(
            missing_values=100,
            n_neighbors=5,
            weights="uniform",
            metric=self.zeroOneDistance,
            copy=False,
        )
        imputer.set_output(transform="pandas")
        working_df = df.drop(
            ["Gris ID", "Sår ID"], axis=1
        )  # remove ID columns so we don't use those for distance calculations
        working_df = imputer.fit_transform(
            working_df
        )  # type: pd.DataFrame # NOTE imputer.set_output(transform="pandas") makes the imputer return a proper dataframe, rather than a numpy array
        for column in working_df.columns:
            df[column] = working_df[column]
        logger.info("Imputation done.")
        self.LogValues(df)
        self.df = df

    def LogValues(self, df: pd.DataFrame, value=100) -> None:
        """Finds and logs a specified value in a dataframe for every ocurrence

        Parameters
        ----------
        df : DataFrame
            The DataFrame to search
        value : The value to find and log using the logger. Default is 100 to help find missing values
        """
        logger.info(f"Checking for {value}...")
        count = 0
        last_index = -1
        rows = 0
        for index, row in df.iterrows():
            for label in df.columns.values:
                if row[label] == value:
                    logger.info(
                        f"Found {value} in {label} at Pig ID {row['Gris ID']}, Wound ID {row['Sår ID']}, Day {row['Dag']} (Internal Index {index})."
                    )
                    count += 1
                    if index != last_index:
                        rows += 1
                        last_index = index
        logger.info(
            f"Counted {count} occurences of {value} in {rows} rows out of {len(df.index)}."
        )

    def minMaxNormalization(self, feature: str) -> None:
        """Uses min-max normalization on a single feature

        Parameters
        ----------
        feature : str
            The feature to be normalized
        """
        self.df[feature] = (self.df[feature] - self.df[feature].min()) / (
            self.df[feature].max() - self.df[feature].min()
        )

    def swapValues(self, attribute, value1, value2) -> None:
        """Swap all instances of value1 and value2 in attribute

        Parameters
        ----------
        attribute : str
            Name of the attribute to swap values in
        value1 : float
            First value
        value2 : float
            Second value
        """
        i = 0
        for value in self.df[attribute]:
            if value == value1:
                self.df.loc[self.df.index[i], attribute] = value2
            elif value == value2:
                self.df.loc[self.df.index[i], attribute] = value1
            i += 1
