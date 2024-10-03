import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing import DataProcessor, Dataset
from modules.visualization import Plotter


class ExudateTypePlotter:
    def __init__(self) -> None:
        dp = DataProcessor(Dataset.REGS)
        dp.deleteNaN()
        self.df = dp.getDataFrame()

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
