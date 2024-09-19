# Include modules for barplot, historgram etc
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt  # Plotting

plt.close("all")  # closes all currently active figures


class AccuracyPlotter:
    def __init__(self) -> None:
        pass

    def barPlot(self, x: ArrayLike | str, y1: ArrayLike, y2: ArrayLike) -> None:
        plt.figure()
        plt.bar(x, y1, color="r")
        plt.bar(x, y2, color="b")
        plt.show()
