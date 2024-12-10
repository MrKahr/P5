import json
import os
from typing import Any
from modules.config.utils.config_batch_processor import ConfigBatchProcessor
from modules.logging import logger
from modules.tools.types import StrPath


class SummaryImporter:
    @classmethod
    def importSummaries(cls, summary_folder: StrPath, keys: list[str]) -> dict:
        configs = ConfigBatchProcessor.getBatchConfigs(summary_folder)
        summaries = {}
        for config in configs:
            with open(config, "r", encoding="utf-8") as file:
                loaded_summary = json.loads(file.read())

            filename = os.path.split(config)[1]
            summaries |= {filename: {}}
            for key in keys:
                try:
                    summaries[filename] |= {key: loaded_summary[key]}
                except KeyError:
                    logger.error(f"Key '{key}' not found in file '{config}'")
        return summaries

    @classmethod
    def processSummaries(cls, summaries: dict, key: str) -> Any:
        for summary in summaries:
            yield summaries[summary][key]
