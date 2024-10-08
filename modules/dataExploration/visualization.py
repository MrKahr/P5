# Include modules for barplot, historgram etc
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

plt.close("all")  # closes all currently active figures


class Plotter:
    def _preprocess(
        self,
        df: pd.DataFrame,
        attribute_x: str,
        attribute_y: str,
        show_percentage: bool,
    ) -> tuple[ArrayLike, ArrayLike, dict]:
        """Extract x/y attributes from the ``df`` dataframe and prepare them for plotting.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing some data to be plotted.
            NOTE: This data MUST be sanitized beforehand.

        attribute_x : str
            The variable to display on the x-axis.

        attribute_y : str
            The variable to display on the y-axis.

        show_percentage : bool
            Show the values of ``attribute_y`` as percentage of total.

        Returns
        -------
        tuple[ArrayLike, ArrayLike, dict]
            Returns a tuple of values, where:
                [0]: NDArray of unique values for ``attribute_x``.
                [1]: NDArray of unique values for ``attribute_y``.
                [2]: dict containing all observations of ``attribute_y`` for each ``attribute_x``.
        """
        # Get all days with data
        unique_x_values = np.sort(df[attribute_x].unique())
        # Get the categorical values of the attribute_y
        unique_y_values = np.sort(df[attribute_y].unique())
        # Initialize data structure for values
        count_y_by_x = {category: [] for category in unique_y_values}

        # Fill out data structure with percentage distribution for each attribute_y
        try:
            for value in unique_x_values:
                x_data = df[df[attribute_x] == value]
                y_counts = x_data[attribute_y].value_counts()
                for category in unique_y_values:
                    y_val = y_counts.get(category, 0)  # Default to 0
                    count_y_by_x[category].append(
                        y_val / len(x_data) * 100 if show_percentage else y_val
                    )
        except ZeroDivisionError:
            print(f"Empty dataset for '{attribute_x}'")
            raise
        return (unique_x_values, unique_y_values, count_y_by_x)

    def _addLabels(
        self,
        ax: Axes,
        attribute_x: str,
        attribute_y: str,
        show_percentage: bool,
        sep: str,
        labels: list[str],
        unique_y_values: ArrayLike,
    ) -> None:
        """Add a labels to the plot and the plot legend.

        Parameters
        ----------
        ax : Axes
            Matplotlib axis class which builds the plot (or something like that)
        attribute_x : str
            The variable to display on the x-axis.

        attribute_y : str
            The variable to display on the y-axis.

        show_percentage : bool
            Show the values of ``attribute_y`` as percentage of total.

        sep : str
            Separate ``labels`` from the unique values of ``attribute_y``. By default ``" "``.

        labels : list[str]
            A label for each unique value of ``attribute_y``.

        unique_y_values : ArrayLike
            The unique values of ``attribute_y``.
        """
        ax.set_ylabel(
            "Percentage in category" if show_percentage else "Observations in category"
        )
        ax.set_xlabel(attribute_x)
        ax.set_title(f"Presence of '{attribute_y}' by '{attribute_x}'")
        ax.legend(
            (
                [f"{val}{sep}{label}" for val, label in zip(unique_y_values, labels)]
                if labels
                else unique_y_values
            ),
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Categories",
        )

    # TODO: Improve or remove
    def barPlot(self, label: list[str] | str, y1: ArrayLike, y2: ArrayLike) -> None:
        plt.figure()
        plt.bar(label, y1, color="r")
        plt.bar(label, y2, color="b")
        plt.show(block=True)

    def stackedBarPlot(
        self,
        dataframe: pd.DataFrame,
        attribute_x: str,
        attribute_y: str,
        show_percentage: bool = True,
        bar_width: int = 2.5,
        labels: Optional[list[str]] = None,
        sep: str = " ",
    ) -> None:
        """Create a stacked bar plot

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing some data to be plotted.
            NOTE: This data MUST be sanitized beforehand.

        attribute_x : str
            The variable to display on the x-axis.

        attribute_y : str
            The variable to display on the y-axis.

        show_percentage : bool
            Show the values of ``attribute_y`` as percentage of total.

        bar_width : int
            The width of the bars. By default ``2.5``.

        labels : list[str], optional
            Custom labels to add to the legend of the plot. By default ``None``.
            NOTE: The amount of labels will be truncated to match the amount of unique values present for ``attribute_y``.

        sep : str
            Separate ``labels`` from the unique values of ``attribute_y``. By default ``" "``.
        """
        # Sift through dataset and extract unique values for *attribute_x* and *attribute_y*
        unique_x_values, unique_y_values, count_x_by_y = self._preprocess(
            dataframe, attribute_x, attribute_y, show_percentage
        )
        fig, ax = plt.subplots(layout="constrained")
        bottom = np.zeros(
            len(unique_x_values)
        )  # The buttom of the stacked bars (used to stack them on top of eachother)
        width = bar_width / len(unique_y_values)  # The width of the bars

        for value, counts in count_x_by_y.items():
            p = ax.bar(
                unique_x_values,
                counts,
                width,
                label=value,
                bottom=bottom,
                align="center",
            )
            bottom += counts

        # Add labels
        self._addLabels(
            ax, attribute_x, attribute_y, show_percentage, sep, labels, unique_y_values
        )
        plt.show()

    def groupedBarPlot(
        self,
        dataframe: pd.DataFrame,
        attribute_x: str,
        attribute_y: str,
        show_percentage: bool = True,
        bar_width: int = 0.75,
        labels: Optional[list[str]] = None,
        sep: str = " ",
    ):
        """Create a grouped bar plot.
        Based on: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing some data to be plotted.
            NOTE: This data MUST be sanitized beforehand.

        attribute_x : str
            The variable to display on the x-axis.

        attribute_y : str
            The variable to display on the y-axis.

        show_percentage : bool
            Show the values of ``attribute_y`` as percentage of total.

        bar_width : int
            The width of the bars. By default ``0.75``.

        labels : list[str], optional
            Custom labels to add to the legend of the plot. By default ``None``.
            NOTE: The amount of labels will be truncated to match the amount of unique values present for ``attribute_y``.

        sep : str
            Separate ``labels`` from the unique values of ``attribute_y``. By default ``" "``.
        """
        unique_x_values, unique_y_values, count_x_by_y = self._preprocess(
            dataframe, attribute_x, attribute_y, show_percentage
        )

        x = np.arange(len(unique_x_values))  # The label locations
        width = bar_width / len(unique_y_values)  # The width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout="constrained")  # Create the plot

        for value, counts in count_x_by_y.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, counts, width, label=value)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add labels
        self._addLabels(
            ax, attribute_x, attribute_y, show_percentage, sep, labels, unique_y_values
        )
        ax.set_xticks(x + width * (len(unique_y_values - 1)) / 2, unique_x_values)
        plt.show()

    def boxPlot(self, dataFrame: pd.DataFrame, colName: str, boxWidth=0.5) -> None:
        fig, ax = plt.subplots()

        plt.boxplot(dataFrame[colName], widths=boxWidth, notch=True)

        # Add labels
        ax.set_xlabel(colName)
        ax.legend(colName, loc="upper right")

        plt.show()

    def scatterPlot(
            self,
            dataFrame: pd.DataFrame,
            attribute_x: str,
            attribute_y: str,
            colName: str,
        ) -> None:

        fig, ax = plt.subplots()

        # sort variables into group
        groups= np.sort(dataFrame[colName].unique())
        group_xMean = [] # Lists to save the mean variables and groups
        group_yMean = []
        group_label = []

        for group in groups:
            p = dataFrame[dataFrame[colName] == group] # Get all variables within current group and plot group
            ax.scatter(p[attribute_x], p[attribute_y], label=group)

            attribute_xMean = np.mean(p[attribute_x]) # Get mean for current group
            attribute_yMean = np.mean(p[attribute_y])

            group_xMean.append(attribute_xMean) # Save mean for current group in lists
            group_yMean.append(attribute_yMean)
            group_label.append(int(group)) # Save groups for labeling of coordinates

        
        ax.plot(group_xMean, group_yMean, label=group_label, marker='.', c="black")

        # Add labels to coordinates
        for i, txt in enumerate(group_label):
            ax.annotate(txt, (group_xMean[i], group_yMean[i]))

        # Add labels
        ax.set_xlabel(attribute_x)
        ax.set_ylabel(attribute_y)
        ax.set_title(f"Relation between '{attribute_x}' and '{attribute_y}'")
        ax.legend(loc='lower right', title=colName)
        ax.grid(True)

        plt.show()
    def plotContinuous(self, dataFrame, y1,y2,x):
        """ plots continuous variables"""
        df = dataFrame
        plt.figure()
        ax = plt.subplot()
        ax.set_ylabel("Sårrand/Sårmidte (cm)")
        ax.set_xlabel("Dag")
        for color in ['tab:blue', 'tab:orange']:
            if (color == 'tab:blue'):
                ax.scatter(df[x], df[y1],s= 120.0, c=color, label=y1,
                    alpha=1,linewidths=0.2, edgecolors='none')
            else:
                ax.scatter(df[x], df[y2],s= 70.0, c=color, label=y2,
                    alpha=1,linewidths=0.2, edgecolors='black')
        uniquedays = np.sort(df[x].unique())
        y1MeanValues = []
        y2MeanValues = []
        for day in uniquedays:
            daydata = df[df[x] == day]
            y1Mean = np.mean(daydata[y1])
            y1MeanValues.append(y1Mean)
            y2Mean = np.mean(daydata[y2])
            y2MeanValues.append(y2Mean)
        #ax.plot(uniquedays,y1MeanValues, 'o-',label=y1)
        #ax.plot(uniquedays,y2MeanValues, 'o-',label=y2)
        ax.legend()

        plt.show()
