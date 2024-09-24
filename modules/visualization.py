# Include modules for barplot, historgram etc
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt  # Plotting
import numpy as np

plt.close("all")  # closes all currently active figures
# plt.ion()


class AccuracyPlotter:
    def barPlot(self, label: list[str] | str, y1: ArrayLike, y2: ArrayLike) -> None:
        plt.figure()
        plt.bar(label, y1, color="r")
        plt.bar(label, y2, color="b")
        plt.show(block=False)

# Plot attribute distribution over time
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
def groupedBarPlot(attribute, dataFrame):
    # Get all days with data
    uniqueDays = np.sort(dataFrame["Dag"].unique())
    # Get the categorical values of the attribute
    uniqueValues = np.sort(dataFrame[attribute].unique())

    # Initialize data structure for values
    categoryCountsByDays = {category: [] for category in uniqueValues}

    # Fill out data structure with percentage distribution for each day
    for day in uniqueDays:
        dayData = dataFrame[dataFrame["Dag"] == day]
        categoryCounts = dayData[attribute].value_counts()
        for value in uniqueValues:
            if len(dayData) == 0: continue # Don't divide by zero
            categoryCountsByDays[value].append(categoryCounts.get(value, 0) / len(dayData) * 100) # Default to 0

    x = np.arange(len(uniqueDays))  # The label locations
    width = 0.75 / len(uniqueValues)  # The width of the bars
    multiplier = 0

    # Create the plot
    fig, ax = plt.subplots(layout="constrained")

    for value, counts in categoryCountsByDays.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, counts, width, label=value)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add labels
    ax.set_ylabel("Percentage in category")
    ax.set_xlabel("Day")
    ax.set_title(f"Presence of \"{attribute}\" by day")
    ax.set_xticks(x + width * (len(uniqueValues - 1)) / 2, uniqueDays)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Categories")

    plt.show()