# Include modules for barplot, historgram etc
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt  # Plotting

plt.close("all")  # closes all currently active figures


class ContinuousPlotter:
    def continousPlot(self, y1: ArrayLike, y2:ArrayLike, x1: ArrayLike) -> None:
        plt.figure()
        plt.scatter(x1,y1)
        plt.scatter(x1,y2)
        plt.show()
