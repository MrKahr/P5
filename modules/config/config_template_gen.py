from typing import Self


class ConfigTemplate(object):
    """Singleton that defines as configuration template for the project.
    Note: We added additional params/longer attribute accesses for clarity."""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._template = (
                cls._createTemplate()
            )  # Added field access for clariy - VS-code cares, Python doesn't
        return cls._instance

    def getTemplate(self) -> dict:
        return self._template

    @classmethod
    def _createTemplate(self) -> dict:
        return {
            "General": {
                "loglevel": "DEBUG",
            },
            "DataPreprocessing": {"test1": ""},
            "ModelSelection": {"test2": ""},
            "ModelTraining": {"test3": ""},
            "ModelTesting": {"test4": ""},
            "ModelEvaluation": {"test5": ""},
        }
