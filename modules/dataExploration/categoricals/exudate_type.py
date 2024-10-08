import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataExploration.visualization import Plotter


class ExudateTypePlotter:
    def __init__(self) -> None:
        dp = DataPreprocessor(Dataset.REGS)
        cleaner = DataCleaner(dp.df)
        cleaner.cleanRegs()
        self.df = dp.df

    def plotExudateType(self) -> None:
        p = Plotter()
        p.stackedBarPlot(
            dataframe=self.df,
            attribute_x="Dag",
            attribute_y="Eksudattype",
            show_percentage=False,
            labels=[
                "serøst",
                "mukøst",
                "purulent",
                "hæmorrhagisk",
                "serohæmorrhagisk",
                "seromukøst",
                "seropurulent",
            ],
        )


if __name__ == "__main__":
    op = ExudateTypePlotter()
    op.plotExudateType()
