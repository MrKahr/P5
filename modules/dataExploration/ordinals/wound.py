import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter


class WoundTissueLevelPlotter:
    def __init__(self) -> None:
        dp = DataPreprocessor(Dataset.REGS)
        cleaner = DataCleaner(dp.df)
        cleaner.cleanRegsDataset()
        self.df = dp.df

    # TODO: Move function to more appropriate class and call it here
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
        self.swapValues("Niveau sårvæv", 2, 4)
        self.swapValues("Niveau sårvæv", 3, 4)
        self.swapValues("Niveau sårvæv", 4, 5)

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
            ],
        )


if __name__ == "__main__":
    op = WoundTissueLevelPlotter()
    op.plotWoundTissueLevel()
