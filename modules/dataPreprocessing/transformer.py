import pandas as pd
from sklearn.impute import KNNImputer

from modules.logging import logger


class DataTransformer:
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
                    
                    logger.info(f"Found missing {label} at Pig ID {row["Gris ID"]}, Wound ID {row["Sår ID"]}, Day {day} (Internal Index {index}). Imputing...") # NOTE 'Internal Index' is the index of the row in the dataframe.
                                                                                                                                                                # It's almost the same as in the excel sheet, but not quite, because we remove unecessary or invalid rows.
                    same_day_rows = df[df["Dag"] == day]
                    column = same_day_rows[label]
                    mode = column.mode()[0] # NOTE mode() returns a dataframe, actually. Since we use it for a single column, there is only one value. Indexing the output with 0 gets us that value.
                    
                    logger.info(f"Mode of {label} is {mode}.")
                    if mode == 100: logger.warning("Mode is a missing value! Cannot properly impute!")
                    
                    df.at[index, label] = mode
                    
                    logger.info(f"Replaced missing value with {mode}.")

    # def zeroOneDistance(label1: str, label2: str) -> int:
    #     """A simple implementation of zero-one distance measuring
    # 
    #     Parameters
    #     ----------
    #     label1 : str
    #         a string to compare
    #     label2 : str
    #         the string to compare with
    # 
    #     Returns
    #     -------
    #     int
    #         0 if the labels are the same, otherwise 1
    #     """
    #     return 0 if label1 == label2 else 1
    
    def zeroOneDistance(x, y, missing_values = 100) -> int:
        """An implementation of a zero-one distance metric in a format scikit's KNNImputer can use

        Parameters
        ----------
        x : 1D numerical array
            A representation of a row in the dataset as an array of numbers
        y : 1D numerical array
            A representation of a row in the dataset as an array of numbers
        missing_values :
            What value should be considered missing and ineligible for comparison, by default 100 (np.nan in the official documentation)

        Returns
        -------
        int
            The distance from x to y measured by counting the number of different entries in the two arrays.
        """
        distance = 0
        for index, entry in enumerate(x):  # NOTE enumerate makes the index available
            if entry != y[index] and not (entry == missing_values or y[index] == missing_values): distance += 1
        return distance
    
    def matrixDistance(x, y, missing_values = 100) -> int:
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
        df = self.df
        imputer = KNNImputer(missing_values=100, n_neighbors=5, weights='uniform', metric=self.zeroOneDistance, copy=False)
        df[:] = imputer.fit_transform(df)
