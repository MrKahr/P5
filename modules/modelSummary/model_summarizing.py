from copy import deepcopy
from typing import Any
from modules.logging import logger


class ModelSummary:

    def __init__(self, model_report: dict):
        self._model_report = model_report

    def _roundConvert(self, value: Any, digits: int = 3) -> str:
        if isinstance(value, int):
            return f"{value}"
        try:
            return f"{value:.{digits}f}"
        except TypeError as e:
            return f"{value}"

    def run(self) -> None:
        formatted = ""
        for k, v in deepcopy(self._model_report).items():
            if k == "feature_importances":
                continue
            if isinstance(v, dict):
                for tk, tv in v.items():
                    v[tk] = self._roundConvert(tv)
            formatted += f"\t{k}: {self._roundConvert(v)}\n"

        logger.info(f"Showing model report:\n{formatted}")
