import json
import os
import re
from datetime import datetime
from itertools import cycle
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from modules.config.utils.config_batch_processor import ConfigBatchProcessor
from modules.config.utils.config_read_write import retrieveDictValue
from modules.config.utils.setup_config import SetupConfig
from modules.tools.types import StrPath


class SummaryImporter:
    @classmethod
    def _showFigure(cls, figure: Figure, figure_filename: str) -> None:
        figure_path = Path(SetupConfig().figures_dir)
        os.makedirs(figure_path, exist_ok=True)
        plt.savefig(
            Path(
                figure_path,
                f"{figure_filename}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png",
            )
        )
        plt.close(figure)

    @classmethod
    def plotAccuracyFunctions(
        cls,
        results: dict[str, ArrayLike],
        results_labels: list[str],
        baseline: dict[str, float],
        x_label: str,
        y_label: str = "Accuracy Score",
        plot_title: str = "Accuracy by Score Function",
        legend_name: str = "Score Functions",
        file_name: str = "scorefunctionPlot",
    ) -> None:
        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot naming
        ax.set(
            xlabel=x_label,
            ylabel=y_label,
            title=plot_title,
        )

        # Set reasonable plot limits
        ax.set_ylim(0, 1)
        ax.set_xlim(auto=True)

        colors = cycle(
            mcolors.XKCD_COLORS
        )  # Long list of colors from https://matplotlib.org/stable/gallery/color/named_colors.html
        x = np.arange(len(results_labels))
        for k, v in results.items():
            plt.plot(v)

            y_min = baseline[k]
            v_arr = np.array(v)
            v_yscaled = v_arr + ((v_arr - y_min) * -1)

            # y_min = [baseline[k] for i in x]
            plt.stem(x, v_arr, basefmt=" ", bottom=y_min)
            # plt.vlines(x, y_min, v, colors=next(colors))

            for i in x:
                ax.annotate(
                    text=f"{(v[i] - y_min):.3f}",
                    xy=(i, v[i]),
                    xytext=(i, v_yscaled[i]),
                )
        ax.set_xticks(
            x,
            results_labels,
        )

        # Plot axis
        ax.legend([*results.keys(), "baseline difference"])
        cls._showFigure(fig, file_name)

    @classmethod
    def importSummaries(cls, summary_folder: StrPath) -> dict:
        configs = ConfigBatchProcessor.getBatchConfigs(summary_folder)
        summaries = {}
        for config in configs:
            with open(config, "r", encoding="utf-8") as file:
                loaded_summary = json.loads(file.read())
            summary_id, filename = os.path.splitext(os.path.split(config)[1])[0].split(
                "_", maxsplit=1
            )
            if summary_id not in summaries:
                summaries |= {summary_id: {}}
            summaries[summary_id] |= {filename: loaded_summary}
        return summaries

    @classmethod
    def processSummariesForPlotting(
        cls, summaries: dict, keys: list[str]
    ) -> tuple[dict, dict, list[str]]:
        processed_summaries = {}
        summary_labels = []
        baseline = {}

        for summary_id, sub_summaries in summaries.items():
            for filename, summary in sub_summaries.items():
                model_name = retrieveDictValue(
                    input=summary, key="model", parent_key="ModelSelection"
                )
                is_baseline = bool(
                    re.search(r"baseline", filename, flags=re.IGNORECASE)
                )

                if model_name not in processed_summaries:
                    processed_summaries |= {model_name: {}}

                model_summary = processed_summaries[model_name]
                for key in keys:
                    target_value = retrieveDictValue(summary, key)
                    if is_baseline:
                        baseline |= {model_name: {key: target_value}}

                    if isinstance(target_value, dict):
                        if key not in model_summary:
                            model_summary |= {key: {}}
                        key_model_summary = model_summary[key]

                        for k, v in target_value.items():
                            if k in key_model_summary:
                                key_model_summary[k].append(v)
                            else:
                                key_model_summary |= {k: [v]}
                    else:
                        model_summary[key] = target_value
            summary_labels.append(summary_id)
        return processed_summaries, baseline, summary_labels

    @classmethod
    def plotSummaries(cls, summary_folder: StrPath, keys: list[str]) -> None:
        summaries, baseline, summary_labels = cls.processSummariesForPlotting(
            cls.importSummaries(summary_folder), keys
        )
        for model, summary in summaries.items():
            for key, result in summary.items():
                model_name = "".join([piece.capitalize() for piece in model.split("_")])
                cls.plotAccuracyFunctions(
                    results=result,
                    results_labels=summary_labels,
                    baseline=baseline[model][key],
                    x_label="Experiment",
                    plot_title=f"Accuracy by Score Function\n{model_name}",
                    file_name=f"scoreFunctionPlot_{model_name}",
                )
