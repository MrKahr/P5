from typing import Self

from modules.config.base_config import BaseConfig

from modules.config.templates.grid_template import GridTemplate
from modules.config.utils.setup_config import SetupConfig


class GridConfig(BaseConfig):
    _instance = None

    # This is a singleton class since we only want 1 instance of a GridConfig at all times
    def __new__(cls) -> Self:
        """The config containing parameter grids for use with GridSearch and friends."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                config_name=SetupConfig.grid_config_name,
                config_path=SetupConfig.grid_config_path,
                template=GridTemplate().getTemplate(),
            )
            self._setConfig(self._initConfig())
            self._created = True
