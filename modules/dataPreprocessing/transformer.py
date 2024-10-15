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

    def modeImputation(self) -> None:
        pass

    def minMaxNormalization(self, feature: str) -> None:
        """Uses min-max normalization on a single feature

        Parameters
        ----------
        feature : str
            The feature to be normalized 
        """
        self.df[feature] = (self.df[feature] - self.df[feature].min()) / (self.df[feature].max() - self.df[feature].min())

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