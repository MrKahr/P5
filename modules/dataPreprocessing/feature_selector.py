import pandas as pd


class FeatureSelector:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
