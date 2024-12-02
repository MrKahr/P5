from typing import Self

from modules.config.base_config import BaseConfig

from modules.config.templates.config_template import ConfigTemplate
from modules.config.utils.setup_config import SetupConfig


class Config(BaseConfig):
    _instance = None

    # This is a singleton class since we only want 1 instance of a Config at all times
    def __new__(cls) -> Self:
        """The config containing general parameters for the entire pipeline."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            super().__init__(
                config_name=SetupConfig.config_name,
                config_path=SetupConfig.pipeline_config_path,
                template=ConfigTemplate().getTemplate(),
            )
            self._setConfig(self._initConfig())
            self._created = True
