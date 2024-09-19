# Include modules for barplot, historgram etc
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt  # Plotting

plt.close("all")  # closes all currently active figures


class AccuracyPlotter:
    def barPlot(self, label: list[str] | str, y1: ArrayLike, y2: ArrayLike) -> None:
        plt.figure()
        plt.bar(label, y1, color="r")
        plt.bar(label, y2, color="b")
        plt.show()
