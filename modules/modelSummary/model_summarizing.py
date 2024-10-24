from modules.logging import logger


class ModelSummary:

    def __init__(self):
        pass

    @classmethod
    def run(self) -> None:
        logger.info("Compiling model summary")
        logger.info(f"Report complete. Handing it over")
