import matplotlib.pyplot as plt  # Plotting
import pandas as pd

from modules.dataPreprocessing import DataProcessor


class OrdinalPlotter:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.df = DataProcessor(dataset)

    def plotScab(self) -> None:
        sub_df = self.df[["Dag", "Niveau sårvæv"]]

    def plotWoundTissueLevel(self) -> None:
        # TODO: Reorder ordinals to have a 'natural' progression
        x = self.df["Niveau sårvæv"]
