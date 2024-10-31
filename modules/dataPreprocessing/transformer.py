import pandas as pd
from sklearn.impute import KNNImputer
from numpy.typing import ArrayLike

from modules.config.config import Config
from modules.config.config_enums import ImputationMethod, NormalisationMethod
from modules.logging import logger


class DataTransformer:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # FIXME: float/int label issue when one-hot encoding
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
            one_hot = pd.get_dummies(self.df[variable], prefix=variable)
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
        # replace the missing value
        # repeat
        df = self.df
        impute_count = 0
        for index, row in df.iterrows():
            for label in df.columns.values:
                if row[label] == 100:
                    day = row["Dag"]

                    # logger.info(
                    #     f"Found missing {label} at Pig ID {row['Gris ID']}, Wound ID {row['Sår ID']}, Day {day} (Internal Index {index}). Imputing..."
                    # )  # NOTE 'Internal Index' is the index of the row in the dataframe.
                    # It's almost the same as in the excel sheet, but not quite, because we remove unecessary or invalid rows.
                    same_day_rows = df[df["Dag"] == day]
                    column = same_day_rows[label]
                    mode = column.mode()[
                        0
                    ]  # NOTE mode() returns a dataframe, actually. Since we use it for a single column, there is only one value. Indexing the output with 0 gets us that value.

                    # logger.info(f"Mode of {label} is {mode}.")
                    if mode == 100:
                        logger.warning(
                            "Mode is a missing value! Cannot properly impute!"
                        )
                    else:
                        impute_count += 1

                    df.at[index, label] = mode

                    # logger.info(f"Replaced missing value with {mode}.")
        logger.info(f"Mode imputation replaced {impute_count} missing values")
        self.df = df

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

    # TODO - Dicussing with Emil, if you see me, delete me!
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

    def knnImputation(self, neighbors: int = 5) -> None:
        """
        Imputes missing values using Scikit's KNNImputer.
        Takes no arguments and modifies the dataframe on the class itself.

        Parameters
        ----------
        neighbors : int
            How many nearest neighbors should be considered.
        """
        logger.info(
            f"Using imputation method: KNN. Args: nearest_neighbors={neighbors}"
        )
        df = self.df
        self.logValues(df)
        logger.info("Starting KNN-Imputation.")
        imputer = KNNImputer(
            missing_values=100,
            n_neighbors=neighbors,
            weights="uniform",
            metric=self.zeroOneDistance,
            copy=False,
        )

        # remove ID columns so we don't use those for distance calculations. Errors are ignored so this goes through even if the columns are already gone.
        # NOTE we do this here even if we might have done it earlier to ensure that Pig ID and Wound ID don't affect imputation regardless of whether "DeleteNonfeatures" in the config is true or not.
        # TODO Consider: Maybe we want to use them, actually? Surely a wound would behave similarly to other wounds on the same animal, or similarly to what it has done in the past?
        working_df = df.drop(["Gris ID", "Sår ID"], axis=1, errors="ignore")

        # NOTE this makes the imputer return a proper dataframe, rather than a numpy array
        imputer.set_output(transform="pandas")
        working_df = imputer.fit_transform(working_df)  # type: pd.DataFrame

        for column in working_df.columns:
            df[column] = working_df[column]

        logger.info(
            "Imputation done."
        )  # TODO: Consider logging here how many rows where imputed or similar
        self.logValues(df)  # TODO: Consider removing or logging in a quieter way
        self.df = df

    def logValues(self, df: pd.DataFrame, value: int = 100) -> None:
        """
        Finds and logs a specified value in a dataframe for every ocurrence.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to search.

        value : int
            The value to find and log using the logger.
            Default is `100` to help find missing values.
        """
        # logger.info(f"Checking for {value}...")
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
        logger.info(
            f"Counted {count} occurences of {value} in {rows} rows out of {len(df.index)}."
        )

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

    # REVIEW: What is this method used for?
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

    def run(self) -> pd.DataFrame:
        """
        Runs all applicable transformation methods.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        config = Config()
        if config.getValue("UseTransformer"):

            # One-hot encoding
            if config.getValue("UseOneHotEncoding"):
                self.oneHotEncode(config.getValue("OneHotEncodeLabels"))

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
                    self.knnImputation(config.getValue("KNN_NearestNeighbors"))
                else:
                    logger.warning(
                        f"Undefined imputation method '{imputation_method}'. Skipping"
                    )
            else:
                logger.info("Skipping imputation")

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
