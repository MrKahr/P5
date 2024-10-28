from typing import Any, Optional, Self

from modules.config.config_read_write import (
    insertDictValue,
    loadConfig,
    retrieveDictValue,
    writeConfig,
)
from modules.config.config_template_gen import ConfigTemplate
from modules.config.setup_config import SetupConfig
from modules.logging import logger


class Config:
    _instance = None
    _logger = logger

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._created = False
        return cls._instance

    def __init__(self) -> None:
        if not self._created:
            self._config_name = SetupConfig.config_name
            self._config_path = SetupConfig.app_config_path
            self._template = ConfigTemplate().getTemplate()
            self._config = self._initConfig()
            self._created = True

    def _initConfig(self) -> dict[str, Any]:
        """Loads the config from a file.

        Returns
        -------
        dict[str, Any]
            The loaded config.
        """
        config = loadConfig(
            config_name=self._config_name,
            config_path=self._config_path,
            template=self._template,
        )
        return config

    def getValue(self, key: str, parent_key: Optional[str] = None) -> Any:
        """Get a value from the config dict object.
        Return first value found. If there is no item with `key`.

        Note: the config is usually nested. Thus, the "get" method of a Python dict
        is insufficient to retrieve all values.

        Parameters
        ----------
        key : str
            The key to search for in the config

        Returns
        -------
        Any
            The value of the key if found, else `default`.
        """
        return retrieveDictValue(input=self._config, key=key, parent_key=parent_key)

    def setValue(self, key: str, value: Any, parent_key: Optional[str] = None) -> bool:
        """Update the config dict with `value` using `key`.

        Parameters
        ----------
        key : str
            The key which value should be updated.

        value : Any
            The value to insert.

        Raises
        ------
        KeyError
            If the key was not found in the config.
        """
        try:
            insertDictValue(self._config, key, value, parent_key)
            writeConfig(self._config, self._config_path)
        except KeyError:
            self._logger.error(
                f"Failed to update config with '{value}' using key '{key}'."
            )
