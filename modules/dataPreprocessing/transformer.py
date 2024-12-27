from typing import Callable
from numpy import ndarray
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from numpy.typing import ArrayLike

from modules.config.pipeline_config import PipelineConfig
from modules.config.utils.config_enums import (
    DiscretizeMethod,
    DistanceMetric,
    ImputationMethod,
    NormalisationMethod,
)
from modules.logging import logger


class DataTransformer:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

        # Define distance matrices. Formatting is turned off for this part so the matrices don't get made into wierd shapes
        # fmt: off
        self._dag_matrix = [
            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
            [ 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34],
            [ 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33],
            [ 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
            [ 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
            [ 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            [ 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
            [ 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
            [ 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
            [ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
            [11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
            [12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
            [13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22],
            [14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21],
            [15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20],
            [16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19],
            [17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18],
            [18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17],
            [19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16],
            [20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15],
            [21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14],
            [22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13],
            [23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12],
            [24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
            [25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
            [26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            [28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
            [29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
            [30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
            [31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
            [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3],
            [33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
            [34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1],
            [35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]
        self._niveau_sårvæv_matrix = [
            [0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0],
        ]
        self._sårskorpe_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self._granulationsvæv_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self._epithelialisering_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self._kontraktion_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self._hyperæmi_matrix = [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
        self._ødem_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self._eksudat_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        self._eksudattype_matrix = [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ]
        self._infektionsniveau_matrix = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ]
        self._sårrand_matrix = [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ]
        self._midte_matrix = [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ]
        # fmt: on

    def oneHotEncode(self, labels: list[str]) -> None:
        """
        One-hot encode one or more categorical attributes, selected by `labels`.

        Based on https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

        Parameters
        ----------
        labels : list[str]
            A list of the column labels to one-hot encode.
        """
        for variable in labels:
            one_hot = pd.get_dummies(self.df[variable], prefix=variable, dtype="int")
            self.df.drop(variable, inplace=True, axis=1)
            self.df = self.df.join(one_hot)
        size = len(labels)
        logger.info(
            f"One-hot encoded {size} feature{"s" if size != 1 else ""}: {", ".join(labels)}"
        )

    def modeImputationByDay(self) -> None:
        """
        Imputes missing values by replacing them with the most common value for the day where the value is missing.
        Takes no arguments and modifies the Dataframe on the class itself.
        """
        # for this method
        # get dataset
        # for every row
        # if there is a missing value, find its day
        # find the most common value in that column for that day
        # replace the missing value (if the most common value is 100, replace with the most common value in the column)
        # repeat
        df = self.df
        working_df = (
            df.copy()
        )  # we will modify a deep copy of the dataset to ensure the order we do things won't affect the result
        impute_count = 0
        fallback_count = 0
        for index, row in df.iterrows():
            for label in df.columns.values:
                # get a fallback-value: the mode of the feature
                # NOTE mode() of a series returns another series, actually. Since there can be multiple modes. Indexing the output with 0 gets us one of those modes.
                feature_column = df[label]  # type: pd.Series
                fallback_value = feature_column[~feature_column.isin[100]].mode()[0]
                logger.info(
                    f'Fallback value for imputation of "{label}" is {fallback_value}.'
                )
                if row[label] == 100:
                    day = row["Dag"]

                    same_day_rows = df[df["Dag"] == day]
                    day_column = same_day_rows[label]  # type: pd.Series
                    mode = day_column.mode()[0]
                    if mode == 100:
                        df.at[index, label] = fallback_value
                        fallback_count += 1

                    impute_count += 1

                    working_df.at[index, label] = mode

        logger.info(
            f"Mode imputation replaced {impute_count} missing values and had to use a fallback value {fallback_count} times."
        )
        self.df = working_df  # we're done. The working df now has all the missing values replaced and is good to go

    def zeroOneDistance(
        self, x: ArrayLike, y: ArrayLike, *args, missing_values: int = 100
    ) -> int:
        """
        An implementation of a zero-one distance metric in a format scikit's KNNImputer can use.

        Parameters
        ----------
        x : 1D Array
            A representation of a row in the dataset as an array of numbers.

        y : 1D Array
            A representation of a row in the dataset as an array of numbers.

        missing_values : int
            The value which should be considered missing and ineligible for comparison, by default 100.

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
        self, x: ArrayLike, y: ArrayLike, *args, missing_values: int = 100
    ) -> int:
        """
        An implementation of a distance metric in a format scikit's KNNImputer can use.
        This one uses distance matrices for each variable.

        Parameters
        ----------
        x : 1D Array
            A representation of a row in the dataset as an array of numbers.

        y : 1D Array
            A representation of a row in the dataset as an array of numbers.

        missing_values : int
            The value which should be considered missing and ineligible for comparison, by default 100.

        Returns
        -------
        int
            The distance from x to y measured by counting the number of different entries in the two arrays.
        """

        # this is a list of the features represented by the two arrays we got as arguments
        labels = self.df.columns.values

        distance = 0
        for index, entry in enumerate(x):  # NOTE enumerate makes the index available
            x_value = int(entry)
            y_value = int(y[index])
            if (
                x_value == missing_values or y_value == missing_values
            ):  # skip distance calculation for a feature if a value is missing
                continue
            try:
                # look up the distance from the x-value to the y-value in the distance matrix corresponding to the feature we're at
                match labels[index]:
                    case "Dag":
                        distance += self._dag_matrix[x_value][y_value]
                    case "Niveau sårvæv":
                        distance += self._niveau_sårvæv_matrix[x_value][y_value]
                    case "Sårskorpe":
                        distance += self._sårskorpe_matrix[x_value][y_value]
                    case "Granulationsvæv":
                        distance += self._granulationsvæv_matrix[x_value][y_value]
                    case "Epithelialisering":
                        distance += self._epithelialisering_matrix[x_value][y_value]
                    case "Kontraktion":
                        distance += self._kontraktion_matrix[x_value][y_value]
                    case "Hyperæmi":
                        distance += self._hyperæmi_matrix[x_value][y_value]
                    case "Ødem":
                        distance += self._ødem_matrix[x_value][y_value]
                    case "Eksudat":
                        distance += self._eksudat_matrix[x_value][y_value]
                    case "Eksudattype":
                        distance += self._eksudattype_matrix[x_value][y_value]
                    case "Infektionsniveau":
                        distance += self._infektionsniveau_matrix[x_value][y_value]
                    case "Sårrand (cm)":
                        distance += self._sårrand_matrix[x_value][y_value]
                    case "Midte (cm)":
                        distance += self._midte_matrix[x_value][y_value]
                    case _:  # default
                        # handling unspecified labels with zero-one distance
                        # a boolean true is equal to 1, false is equal to 0
                        distance += x_value != y_value
            except (
                IndexError
            ):  # if try to access an entry in a distance matrix that doesn't exist, we end up here
                logger.warning(
                    f"No entry in distance matrix for {labels[index]} at {x_value}, {y_value}. Skipping distance calculation for those values."
                )
        return distance

    def knnImputation(
        self,
        distance_metric: Callable[[ArrayLike, ArrayLike, int], int],
        neighbors: int = 5,
    ) -> None:
        """
        Imputes missing values using Scikit's KNNImputer.
        Takes no arguments and modifies the dataframe on the class itself.

        Parameters
        ----------
        neighbors : int
            How many nearest neighbors should be considered.
        """
        df = self.df
        initial_count = self.countValues(df)
        imputer = KNNImputer(
            missing_values=100,
            n_neighbors=neighbors,
            weights="distance",
            metric=distance_metric,
            copy=False,
        )

        # remove ID columns so we don't use those for distance calculations. Errors are ignored so this goes through even if the columns are already gone.
        # NOTE we do this here even if we might have done it earlier to ensure that Pig ID and Wound ID don't affect imputation regardless of whether "DeleteNonfeatures" in the config is true or not.
        working_df = df.drop(["Gris ID", "Sår ID"], axis=1, errors="ignore")

        # NOTE this makes the imputer return a proper dataframe, rather than a numpy array
        imputer.set_output(transform="pandas")
        working_df = imputer.fit_transform(working_df)  # type: pd.DataFrame

        for column in working_df.columns:
            df[column] = working_df[column]

        replaced_count = initial_count - self.countValues(df)
        logger.info(
            f"KNN Imputation replaced {replaced_count} missing value{"s" if replaced_count != 1 else ""}"
        )
        self.df = df

    def countValues(self, df: pd.DataFrame, value: int = 100) -> None:
        """
        Count every ocurrence of `value` in `df`.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to search.

        value : int
            The value search for.
            Default is `100` to help find missing values.
        """
        count = 0
        last_index = -1
        rows = 0
        for index, row in df.iterrows():
            for label in df.columns.values:
                if row[label] == value:
                    # logger.info(
                    #     f"Found {value} in {label} at Pig ID {row['Gris ID']}, Wound ID {row['Sår ID']}, Day {row['Dag']} (Internal Index {index})."
                    # )
                    count += 1
                    if index != last_index:
                        rows += 1
                        last_index = index
        return count

    def minMaxNormalization(self, feature: str) -> None:
        """
        Uses min-max normalization on a single feature.

        Parameters
        ----------
        feature : str
            The feature to be normalized.
        """
        self.df[feature] = (self.df[feature] - self.df[feature].min()) / (
            self.df[feature].max() - self.df[feature].min()
        )

    def swapValues(self, feature: str, value1: float, value2: float) -> None:
        """
        Swap all instances of `value1` and `value2` in `feature`.

        Parameters
        ----------
        feature : str
            Name of the feature to swap values in.

        value1 : float
            First value.

        value2 : float
            Second value.
        """
        i = 0
        for value in self.df[feature]:
            if value == value1:
                self.df.loc[self.df.index[i], feature] = value2
            elif value == value2:
                self.df.loc[self.df.index[i], feature] = value1
            i += 1

    def discretizeWithChiMerge(
        self,
        value_column_name: str,
        class_column_name: str = "Dag",
        merge_when_below: float = np.inf,
        desired_intervals: int = -1,
    ) -> list[float]:
        """
        An implementation of the ChiMerge algorithm that returns a list of interval's lower bounds given a dataframe,
        a column of values to discretize, and a column to consider as classes(labels)

        Parameters
        ----------
        class_column_name : str
            The name of the dataframe column that the algorithm should consider as holding class labels
        value_column_name : str
            The name of the dataframe column that holds the values to discretize
        merge_when_below: float
            Intervals will only be merged when their chi-square value is below this number
        desired_intervals: int
            When the number of intervals is equal to this, no more intervals will be merged

        Returns
        -------
        list[float]
            A list of numbers that specify the lower bounds (inclusive) of non-overlapping intervals for the values to be categorized into
        """
        logger.info(
            f"Preparing discretization of column '{value_column_name}' with class column '{class_column_name}'"
        )

        # get values to discretize given some column name and sort them from smallest to biggest. Also handle situation where cleaner has not been used
        values = self.df[value_column_name].dropna().to_numpy()  # type: ndarray
        values.sort()
        # values of 100 are undefined, so we remove those before moving on
        values = values[values != 100]

        # get unique values for initial interval lower bounds. Using unique() instead of copy() saves a lot of iterations, as ChiMerge will always merge identical intervals
        lower_bounds = np.unique(values)

        # find distinct classes given some column name ("Dag", for example)
        classes = self.df[class_column_name].dropna().unique()

        # chi_squares[i] holds the chi-square of the intervals with the lower bounds lower_bounds[i] and lower_bounds[i+1]
        chi_squares = np.empty(lower_bounds.size - 1)

        # initialise this to get the while loop going
        minimum_chi_square = -np.inf
        current_range = range(lower_bounds.size - 1)
        # NOTE we're working with pairs of lower bounds, so we stop iterating at the second-to-last index
        # the first current_range covers all interval pairs to get a value for all of them. Subsequent ones will cover only intervals affected by lower bounds' removals

        # NOTE see section on ChiMerge for how the following works
        # (optimization proposed by Kerber: Only recalculate values for affected intervals i.e. lower_bounds[i-1] and lower_bounds[i], and lower_bounds[i] and lower_bounds[i+1])

        logger.info(
            f"Running ChiMerge with {len(values)} values and {len(classes)} classes"
        )

        while (
            lower_bounds.size > 1
            and minimum_chi_square < merge_when_below
            and lower_bounds.size != desired_intervals
        ):
            # calculate chi-square for all pairs of adjacent intervals.
            for index in current_range:

                chi_square = 0
                for i in range(2):
                    # define the bounds of the current interval
                    current_lower_bound = lower_bounds[index + i]
                    # if the upper bound does not exist, it stays as infinity
                    current_upper_bound = np.inf
                    if index + i + 1 < lower_bounds.size:
                        current_upper_bound = lower_bounds[index + i + 1]

                    for j in classes:
                        # find the number of examples in class j
                        C_j = self.df[self.df[class_column_name] == j].shape[0]

                        # find the number of examples in the current interval by counting the how many Trues there are in the series returned by between() with sum()
                        R_i = (
                            self.df[value_column_name]
                            .between(
                                current_lower_bound,
                                current_upper_bound,
                                inclusive="left",
                            )
                            .sum()
                        )
                        # find the number of examples of class j in the current interval
                        A_ij = (
                            (self.df[self.df[class_column_name] == j])[
                                value_column_name
                            ]
                            .between(
                                current_lower_bound,
                                current_upper_bound,
                                inclusive="left",
                            )
                            .sum()
                        )
                        # find the expected value of the number of examples of class j in the current interval
                        E_ij = (R_i * C_j) / values.size
                        chi_square += (A_ij - E_ij) ** 2 / E_ij

                chi_squares[index] = chi_square

            # we're done finding chi_squares. Now merge the two intervals with the smallest value
            minimum_chi_square = min(chi_squares)
            index = np.where(chi_squares == minimum_chi_square)[0][0]
            # intervals can be merged by deleting the largest of the two lower bounds: Merging [a,b) and [b,c) gives [a,c)!
            chi_squares = np.delete(chi_squares, index)
            lower_bounds = np.delete(lower_bounds, index + 1)

            # we don't want to recalculate chi-squared values for all interval pairs, so we update current_range to only include the affected intervals
            # 0 1 2 3 -> 0 1 2 3
            # a b c d    a b d
            # see above what happens when we delete interval bound c to merge its interval with interval bound b's interval
            # we're interested in the pairs a & b and b & d. That's the interval represented by lower_bounds[index-1] & lower_bounds[index] and lower_bounds[index] & lower_bounds[index+1]
            # but lower_bounds[index+1] does not exist in the case where index = lower_bounds.size - 1 i.e. when we have merged the last two intervals
            # and lower_bounds[index-1] does not exist in the case where index = 0 i.e. when we have merged the first two intervals
            # for the general case, our new current_range will be [index-1, index] (remember that we check intervals with bounds at index and index + 1 each iteration)
            # if index = lower_bounds.size - 1, our range will be [index-1]
            # if index = 0, our range will be [index]

            if index == (lower_bounds.size - 1):
                # we have just merged the last two intervals
                current_range = [index - 1]
            elif index == 0:
                # we have just merged the first two intervals
                current_range = [index]
            else:
                # we have not merged the first or last two intervals
                current_range = [index - 1, index]

        logger.info(
            f"Intervals for '{value_column_name}' generated. Interval bounds are {lower_bounds.tolist()}"
        )

        # when we're done merging intervals, return the list of lower bounds
        return lower_bounds

    def discretizeNaively(
        self,
        column_name: str,
        desired_intervals: int = 1,
    ) -> list[float]:
        """
        A naive discretization method that splits the values in the given column into a number of intervals
        where each interval has the same length.

        Parameters
        ----------
        column_name : str
            The name of the dataframe column that holds the values to discretize
        desired_intervals : int
            The number of intervals to generate

        Returns
        -------
        list[float]
            A list of numbers that specify the lower bounds (inclusive) of non-overlapping intervals for the values to be categorized into

        Raises
        ------
        ValueError
            If the number of desired intervals is 0 or less, no intervals can be generated
        """
        logger.info(f"Preparing discretization of column '{column_name}'")

        values = self.df[column_name].to_numpy()
        # if desired intervals is 0 or less, we can't split the column into any intervals!
        if desired_intervals < 1:
            raise ValueError("Desired intervals must be 1 or more")

        # values of 100 are undefined, so we remove those before moving on
        values = values[values != 100]

        logger.info(f"Running naive discretization with {len(values)} values")

        # find out how big each step is when we need desired_intervals intervals
        step = (values.max() - values.min()) / desired_intervals
        lower_bounds = np.empty(desired_intervals)
        for i in range(desired_intervals):
            lower_bounds[i] = values.min() + (step * i)

        logger.info(
            f"Intervals for '{column_name}' generated. Interval bounds are {lower_bounds.tolist()}"
        )

        return lower_bounds

    def assignIntervals(
        self,
        column_name: str,
        lower_bounds: list[float],
    ) -> None:
        """
        Replaces values in a given column with numbers representing the interval they fit into

        Parameters
        ----------
        lower_bounds : list[float]
            A list of numbers where each number represents the lower bound of an interval.
            Note that intervals expressed like this never overlap, and exclude their upper bound, which is the lower bound for the next interval.
            The list should be sorted from smallest to biggest.
        column_name: str
            The name of the column whose values should be replaced
        replace_blacklist: list[float]
            A list of numbers that may or may not occur in the column and shouldn't be replaced
        """
        logger.info(f"Assigning intervals to values in '{column_name}'")

        def intervalify(
            x: float, lower_bounds: list[float], replace_blacklist: list[float] = [100]
        ) -> int or float:  # type: ignore
            """Helper function for assignIntervals that maps a value to its interval

            Parameters
            ----------
            x : float
                The value to map
            lower_bounds : list[float]
                A list of non-overlapping intervals' lower bounds
            replace_blacklist : list[float]
                A list of values that should not be replaced. For example missing values.
            Returns
            -------
            int or float
                The index of the interval that x fits into, or x, if x is in the blacklist
            """
            for i in range(len(lower_bounds)):
                if x in replace_blacklist:
                    return x
                upper_bound = np.inf
                if (i + 1) < len(lower_bounds):
                    upper_bound = lower_bounds[i + 1]
                if lower_bounds[i] <= x < upper_bound:
                    return i

        series_to_modify = self.df[column_name]
        self.df[column_name] = series_to_modify.apply(
            intervalify, lower_bounds=lower_bounds
        )

        logger.info(f"Discretization of '{column_name}' complete")

    def run(self) -> pd.DataFrame:
        """
        Runs all applicable transformation methods.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        config = PipelineConfig()
        if config.getValue("UseTransformer"):

            # Imputation
            imputation_method = config.getValue("ImputationMethod")
            if imputation_method != ImputationMethod.NONE.name:
                missing_val_key = "DeleteMissingValues"
                if config.getValue(missing_val_key, "Cleaning"):
                    logger.warning(
                        f"Cannot impute correctly when {missing_val_key} is True. Aborting"
                    )
                elif imputation_method == ImputationMethod.MODE.name:
                    self.modeImputationByDay()
                elif imputation_method == ImputationMethod.KNN.name:
                    metric = None
                    match config.getValue("KNN_DistanceMetric"):
                        case DistanceMetric.ZERO_ONE.name:
                            metric = self.zeroOneDistance
                        case DistanceMetric.MATRIX.name:
                            metric = self.matrixDistance
                    self.knnImputation(metric, config.getValue("KNN_NearestNeighbors"))
                else:
                    logger.warning(
                        f"Undefined imputation method '{imputation_method}'. Skipping"
                    )
            else:
                logger.info("Skipping imputation")

            # Discretization
            discretize_method = config.getValue("DiscretizeMethod", "Transformer")
            if discretize_method == DiscretizeMethod.NONE.name:
                pass
            elif discretize_method == DiscretizeMethod.CHIMERGE.name:
                for column in config.getValue("DiscretizeColumns"):
                    value = config.getValue("ChiMergeMaximumMergeThreshold").get(column)
                    if value == "inf":
                        value = np.inf

                    interval_bounds = self.discretizeWithChiMerge(
                        column,
                        merge_when_below=value,
                        desired_intervals=config.getValue(
                            "DiscretizeDesiredIntervals"
                        ).get(column),
                    )
                    self.assignIntervals(
                        column,
                        lower_bounds=interval_bounds,
                    )
            elif discretize_method == DiscretizeMethod.NAIVE.name:
                for column in config.getValue("DiscretizeColumns"):
                    interval_bounds = self.discretizeNaively(
                        column,
                        desired_intervals=config.getValue(
                            "DiscretizeDesiredIntervals"
                        ).get(column),
                    )
                    self.assignIntervals(
                        column,
                        lower_bounds=interval_bounds,
                    )
            else:
                logger.warning(
                    f"Undefined discretization method '{discretize_method}'. Skipping"
                )

            # One-hot encoding
            if config.getValue("UseOneHotEncoding"):
                self.oneHotEncode(config.getValue("OneHotEncodeLabels"))

            # Normalization
            match config.getValue("NormalisationMethod"):
                case NormalisationMethod.MIN_MAX.name:
                    normalize_features = config.getValue("NormaliseFeatures")
                    logger.info(f"Using normalisation method: min-max")
                    for feature in normalize_features:
                        self.minMaxNormalization(feature)

                    size = len(normalize_features)
                    if size > 0:
                        logger.info(
                            f"Normalized {size} feature{"s" if size !=  1 else ""}: {", ".join(normalize_features)}"
                        )
                    else:
                        logger.warning(
                            f"Failed to normalize features: Feature list is empty!"
                        )
                case NormalisationMethod.NONE.name:
                    logger.info("Skipping normalisation")
                case _:  # default
                    logger.warning("Undefined normalisation method selected. Skipping")
        else:
            logger.info("Skipping data transformation")

        return self.df
