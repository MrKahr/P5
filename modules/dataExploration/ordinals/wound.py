import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter
from modules.dataPreprocessing.transformer import DataTransformer


class WoundTissueLevelPlotter:
    def __init__(self) -> None:
        dp = DataPreprocessor(Dataset.REGS)
        cleaner = DataCleaner(dp.df)
        cleaner.cleanRegsDataset()
        self.df = dp.df



    def plotWoundTissueLevel(self) -> None:
        # Swaps values to the following:
        """ "
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
        transformer = DataTransformer(self.df)
        transformer.swapValues("Niveau sårvæv", 2, 4)
        transformer.swapValues("Niveau sårvæv", 3, 4)
        transformer.swapValues("Niveau sårvæv", 4, 5)

        p = Plotter()
        p.stackedBarPlot(
            dataframe=self.df,
            attribute_x="Dag",
            attribute_y="Niveau sårvæv",
            labels=[
                "under niveau",
                "under og i niveau",
                "i niveau",
                "over og i niveau",
                "over niveau",
                "over og under niveau"
            ],
        )


if __name__ == "__main__":
    op = WoundTissueLevelPlotter()
    op.plotWoundTissueLevel()
