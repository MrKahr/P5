import pandas as pd

from modules.dataPreprocessing.preprocessor import DataPreprocessor


class DataTransformer(DataPreprocessor):
    def __init__(self) -> None:
        super().__init__()

    def oneHotEncoding(self, labels: list[str]) -> None:
        """One-hot encode one or more categorical attributes

        Based on https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

        Parameters
        ----------
        labels : list[str]
            A list of the column labels to one-hot encode
        """
        for variable in labels:
            one_hot = pd.get_dummies(self.df[variable], prefix=variable)
            self.df.drop(variable, inplace=True, axis=1)
            self.df = self.df.join(one_hot)

    # TODO: Implement mode imputation
    def modeImputation(self) -> None:
        pass
