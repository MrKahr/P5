# Include modules for barplot, historgram etc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close("all")  # closes all currently active figures


class Plotter:
    def barPlot(
        self, label: list[str] | str, y1: np.typing.ArrayLike, y2: np.typing.ArrayLike
    ) -> None:
        plt.figure()
        plt.bar(label, y1, color="r")
        plt.bar(label, y2, color="b")
        plt.show(block=True)

    def stackedBarPlot(
        self, dataframe: pd.DataFrame, attribute_x: str, attribute_y
    ) -> None:
        df = dataframe
        # Get all days with data
        uniqueDays = np.sort(df[attribute_x].unique())
        # Get the categorical values of the attribute_y
        uniqueValues = np.sort(df[attribute_y].unique())

        # Initialize data structure for values
        categoryCountsByDays = {category: [] for category in uniqueValues}

        # Fill out data structure with percentage distribution for each attribute_y
        for value in uniqueDays:
            dayData = df[df[attribute_x] == value]
            categoryCounts = dayData[attribute_y].value_counts()

            for value in uniqueValues:
                if len(dayData) == 0:
                    continue  # Don't divide by zero
                categoryCountsByDays[value].append(
                    categoryCounts.get(value, 0) / len(dayData) * 100
                )  # Default to 0

        width = 0.9 / len(uniqueValues)  # The width of the bars

        # Create the plot
        fig, ax = plt.subplots()
        bottom = np.zeros(len(uniqueDays))

        for value, counts in categoryCountsByDays.items():
            p = ax.bar(
                uniqueDays, counts, width, label=value, bottom=bottom, align="center"
            )
            bottom += counts

        # Add labels
        ax.set_ylabel("Percentage in category")
        ax.set_xlabel(attribute_x)
        ax.set_title(f"Presence of '{attribute_y}' by '{attribute_x}'")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Categories")

        plt.show()

    def groupedBarPlot(self, dataFrame, yattribute, xattribute="Dag"):
        """Plot attribute distribution over time.

        Based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        """
        # Get all days with data
        uniqueDays = np.sort(dataFrame[xattribute].unique())
        # Get the categorical values of the attribute
        uniqueValues = np.sort(dataFrame[yattribute].unique())

        # Initialize data structure for values
        categoryCountsByDays = {category: [] for category in uniqueValues}

        # Fill out data structure with percentage distribution for each day
        for day in uniqueDays:
            dayData = dataFrame[dataFrame[xattribute] == day]
            categoryCounts = dayData[yattribute].value_counts()
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
        ax.set_title(f"Presence of \"{yattribute}\" by day")
        ax.set_xticks(x + width * (len(uniqueValues - 1)) / 2, uniqueDays)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Categories")

        plt.show()