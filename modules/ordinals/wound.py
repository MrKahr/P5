import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter


class WoundTissueLevelPlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()

    def plotWoundTissueLevel(self) -> None:
        p = Plotter()
        p.stackedBarPlot(
            dataframe=self.df, attribute_x="Dag", attribute_y="Niveau sårvæv"
        )
        # TODO: Reorder ordinals to have a 'natural' progression
        # TODO: Add labels when reordering is done
        """"
        1: under niveau
        2: i niveau
        3: over niveau
        4: under og i niveau
        5: over og i niveau

        V

        1: under niveau
        2: under og i niveau
        3: i niveau
        4: over og i niveau
        5: over niveau
        """


if __name__ == "__main__":
    op = WoundTissueLevelPlotter()
    op.plotWoundTissueLevel()
