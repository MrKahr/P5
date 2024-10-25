from modules.logging import logger


class ModelSummary:

    def __init__(self, model_report: dict):
        self._model_report = model_report

    def run(self) -> None:
        logger.info(f"Showing model report:\n\t{self._model_report}")
