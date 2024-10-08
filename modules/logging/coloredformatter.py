import logging
from colorama import init


class ColoredFormatter(logging.Formatter):
    init(autoreset=True)
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m',  # Red
        'RESET': '\033[0m'   # Reset colors
    }

    def format(self, record) -> str:
        log_level = record.levelname
        color_start = self.COLORS.get(log_level, self.COLORS['RESET'])
        color_end = self.COLORS['RESET']
        record.levelname = f"{color_start}{log_level}{color_end}"
        return super().format(record)
