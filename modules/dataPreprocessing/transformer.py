import pandas as pd


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
        # TODO for this method
        # get dataset
        # for every row
        # if there is a missing value, find its day
        # find the most common value in that column for that day
        # replace the missing value
        # repeat
        df = self.df
        for row in df.iterrows():
            for label in df.columns.values:
                if row[label] == 100:
                    day = row["Dag"]
                    same_day_rows = df[(df["Dag"] == day)]
                    column = same_day_rows[label]
                    mode = column.mode()
                    df.at(row.index, label) = mode

    def zeroOneDistance(label1: str, label2: str) -> int:
        """A simple implementation of zero-one distance measuring

        Parameters
        ----------
        label1 : str
            a string to compare
        label2 : str
            the string to compare with

        Returns
        -------
        int
            0 if the labels are the same, otherwise 1
        """
        return 0 if label1 == label2 else 1