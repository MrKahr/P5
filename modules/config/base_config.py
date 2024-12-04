from typing import Any, Optional

import modules.config.utils.config_read_write as config_rw
from modules.logging import logger
from modules.tools.types import StrPath


class BaseConfig:
    """Base class for all configs"""

    _logger = logger

    def __init__(self, config_name: str, config_path: StrPath, template: dict) -> None:
        """
        Initiate config wrapper class.

        Remember to call setter for config!
        ----
        config : dict
            The config's underlying dict.

        Parameters
        ----------
        config_name : str
            The name of `config`.

        config_path : StrPath
            The config's location on disk.

        template : dict | None
            The template of `config`.
        """
        self._config_name = config_name
        self._config_path = config_path
        self._config = None  # type: dict[str, Any]
        self._template = template

    def _initConfig(self) -> dict[str, Any]:
        """
        Loads the config from a file.

        Returns
        -------
        dict[str, Any]
            The loaded config.
        """
        config = config_rw.loadConfig(
            config_name=self._config_name,
            config_path=self._config_path,
            template=self._template,
        )
        return config

    def _setConfig(self, config: dict[str, Any]) -> None:
        self._config = config

    def getConfig(self) -> dict[str, Any]:
        """
        Get the config's underlying dict.

        Returns
        -------
        dict[str, dict]
            The config's underlying dict
        """
        return self._config

    def getConfigPath(self) -> StrPath:
        return self._config_path

    def getValue(self, key: str, parent_key: Optional[str] = None) -> Any:
        """
        Get a value from the config dict object.
        Returns first value found.

        Note: the config is usually nested. Thus, the "get" method of a Python dict
        is insufficient to retrieve all values.

        Parameters
        ----------
        key : str
            The key to search for in the config.

        Returns
        -------
        Any
            The value of `key`, if found.

        Raises
        ------
        UnboundLocalError
            If `key` was not found in the config.
        """
        return config_rw.retrieveDictValue(
            input=self._config, key=key, parent_key=parent_key
        )

    def setValue(self, key: str, value: Any, parent_key: Optional[str] = None) -> None:
        """
        Assign `value` to `key` in the config's underlying dict, overwriting any previous value.

        Parameters
        ----------
        key : str
            The key which value should be updated.

        value : Any
            The value to insert.

        Raises
        ------
        KeyError
            If `key` was not found in the config.
        """
        try:
            config_rw.insertDictValue(self._config, key, value, parent_key)
        except KeyError:
            self._logger.error(
                f"Failed to update config with '{value}' using key '{key}' {f"inside the scope of parent key '{parent_key}'" if parent_key is not None else ""}."
            )

    def writeToDisk(self) -> None:
        """
        Saves the config's underlying dict to disk.
        The file is defined in the instance variable `_config_path`.
        """
        config_rw.writeConfig(self._config, self._config_path)
