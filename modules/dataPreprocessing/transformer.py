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
