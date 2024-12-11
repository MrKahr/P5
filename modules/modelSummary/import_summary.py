from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from modules.config.utils.config_batch_processor import ConfigBatchProcessor
from modules.config.utils.setup_config import SetupConfig
from modules.logging import logger
from modules.modelSummary.model_summarizing import ModelSummary
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
        results: dict[str, NDArray],
        results_labels: list[str],
        x_label: str,
        y_label: str = "Accuracy Score",
        plot_title: str = "Accuracy by Score Function",
        legend_name: str = "Score Functions",
        file_name: str = "scorefunctionPlot",
    ) -> None:
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot naming
        ax.set(
            xlabel=x_label,
            ylabel=y_label,
            title=plot_title,
        )

        # Set reasonable plot limits
        ax.set_ylim(0, 1)
        ax.set_xlim(auto=True)

        for k, v in results.items():
            plt.plot(v)

        ax.set_xticks(
            np.arange(len(results_labels)),
            results_labels,
        )

        # Plot axis
        ax.legend(results.keys())
        cls._showFigure(fig, file_name)

    @classmethod
    def importSummaries(cls, summary_folder: StrPath, keys: list[str]) -> dict:
        configs = ConfigBatchProcessor.getBatchConfigs(summary_folder)
        summaries = {}
        for config in configs:
            with open(config, "r", encoding="utf-8") as file:
                loaded_summary = json.loads(file.read())

            filename = os.path.split(config)[1].split("_")[0]
            summaries |= {filename: {}}
            for key in keys:
                try:
                    summaries[filename] |= {key: loaded_summary[key]}
                except KeyError:
                    logger.error(f"Key '{key}' not found in file '{config}'")
        return summaries

    @classmethod
    def processSummariesForPlotting(cls, summaries: dict) -> tuple[dict, list[str]]:
        processed_summaries = {}
        summary_labels = []

        for filename, summary in summaries.items():
            for key, value in summary.items():
                if key not in processed_summaries:
                    processed_summaries |= {key: {}}
                for score_func_name, score in value.items():
                    if score_func_name in processed_summaries[key]:
                        processed_summaries[key][score_func_name].append(score)
                    else:
                        processed_summaries[key] |= {score_func_name: [score]}

            summary_labels.append(filename)
        return processed_summaries, summary_labels

    @classmethod
    def plotSummaries(cls, summary_folder: StrPath, keys: list[str]) -> None:
        summaries, summary_labels = cls.processSummariesForPlotting(
            cls.importSummaries(summary_folder, keys)
        )

        for key, value in summaries.items():
            cls.plotAccuracyFunctions(
                results=value,
                results_labels=summary_labels,
                x_label="Experiments",
            )
