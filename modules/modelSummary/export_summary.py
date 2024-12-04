from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from modules.config.config import Config
from modules.config.grid_config import GridConfig
from modules.config.utils.setup_config import SetupConfig
from modules.logging import logger


class SummaryExporter:

    @classmethod
    def _serialize(cls, input_value: Any) -> Any:
        """
        Serialize objects to JSON.

        Parameters
        ----------
        input_value : Any
            An object to serialize.

        Returns
        -------
        Any
            The serialized object.
        """
        if isinstance(input_value, dict):
            d = {}
            for k, v in input_value.items():
                d[k] = cls._serialize(v)
            return d
        elif isinstance(input_value, np.ndarray):
            input_value = input_value.tolist()

        return input_value

    @classmethod
    def export(
        cls, pipeline_report: dict, batch_number: int, batch_total: int, batch_id: str
    ) -> None:
        """
        Export select entries in the pipeline report as a model summary.
        Includes the configs used to create this summary.

        Parameters
        ----------
        pipeline_report : dict
            The report generated from the pipeline.

        batch_number : int
            Current batch number when running in batch mode.
            Used for numbering the summaries of this batch run.

        batch_total : int
            Total amount of batches in this batch run.

        batch_id : str
            The unique ID of this batch.
            Used to distinguish summaries across batch runs.
        """
        export_folder = Path(SetupConfig.summary_dir, batch_id)
        os.makedirs(export_folder, exist_ok=True)
        logger.info(f"Exporting model summary")

        export_dict = {}
        for k, v in pipeline_report.items():
            if k in [
                "train_accuracies",
                "train_precision",
                "train_recall",
                "test_accuracies",
                "test_precision",
                "test_recall",
                "train_pred_y",
                "test_pred_y",
                "feature_importances",
                "feature_names_in",
            ]:
                export_dict[k] = cls._serialize(v)

        export_dict[SetupConfig.pipeline_config_file.split(".")[1]] = (
            Config().getConfig()
        )
        export_dict[SetupConfig.grid_config_file.split(".")[0]] = (
            GridConfig().getConfig()
        )

        estimator_name = type(pipeline_report["estimator"]).__name__
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{batch_number}_{batch_total}.summary.{estimator_name}.{time}.json"
        with open(Path(export_folder, file_name), "w", encoding="utf-8") as file:
            file.write(json.dumps(export_dict, indent=4))