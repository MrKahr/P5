import logging
import os

from datetime import datetime
from typing import Self

from modules.config.setup_config import SetupConfig
from modules.logging.coloredformatter import ColoredFormatter
from modules.logging.colorcodefilter import ColorCodeFilter


class Logger:
    _instance = None
    app_name = SetupConfig.app_name
    log_dir = SetupConfig.log_dir
    log_format = SetupConfig.log_format
    log_format_color = SetupConfig.log_format_color

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._create_logger("DEBUG")
            cls._instance._create_logger_title()
        return cls._instance

    def _current_datetime(self) -> str:
        # return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return datetime.now().strftime("%Y-%m-%d")

    def _create_logger(self, level) -> logging.Logger:
        self.logger = logging.getLogger(self.app_name)
        self.logger.propagate = False
        self.logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(self.log_format_color)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if not self.log_dir.exists():
            self.log_dir.mkdir()
        file_handler = logging.FileHandler(
            str(self.log_dir) + os.sep + f"{self._current_datetime()}.log",
            encoding="utf-8",
        )
        file_formatter = ColorCodeFilter(self.log_format)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        return self.logger

    def _create_logger_title(self, level="INFO") -> logging.Logger:
        self.logger_title = logging.getLogger(f"{self.app_name}_title")
        self.logger_title.propagate = False
        self.logger_title.setLevel(level)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger_title.addHandler(console_handler)

        if not self.log_dir.exists():
            self.log_dir.mkdir()
        file_handler = logging.FileHandler(
            str(self.log_dir) + os.sep + f"{self._current_datetime()}.log",
            encoding="utf-8",
        )
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger_title.addHandler(file_handler)

        return self.logger_title

    @classmethod
    def writeHeaderToLog(cls) -> None:
        padding = 90
        header = (
            "┌"
            + "─" * padding
            + "┐"
            + "\n"
            + "│"
            + f"Starting application".center(padding, " ")
            + "│"
            + "\n"
            + "└"
            + "─" * padding
            + "┘"
        )
        cls._instance.logger_title.info(f"\n{header}")

    def get_logger(self) -> logging.Logger:
        return self.logger
