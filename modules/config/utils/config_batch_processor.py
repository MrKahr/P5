from copy import deepcopy
import glob
import os
from typing import Any, Generator
from modules.config.utils.setup_config import SetupConfig
from modules.logging import logger
from modules.tools.types import StrPath


class ConfigBatchProcessor:
    _logger = logger

    @classmethod
    def getBatchConfigs(
        cls, folder: StrPath, extensions: list[str] = ["json"]
    ) -> list[StrPath]:
        """
        Recursively find all files matching an extension in `extensions`.

        Parameters
        ----------
        folder : StrPath
            The directory which to search in.

        extensions : list[str], optional
            Match files with these extensions.
            By default `["json"]`.

        Returns
        -------
        list[StrPath]
            A sorted list of files matching `extensions` found in `folder`.
        """
        files = []
        for extension in extensions:
            files.extend(glob.glob(f"{folder}/**/*.{extension}", recursive=True))
        return sorted(files)

    @classmethod
    def getConfigPairsFromBatch(
        cls,
        configs: list[StrPath],
    ) -> Generator[list[StrPath], Any, None]:
        """
        Get a list of matching configs for use in a pipeline.

        Parameters
        ----------
        configs : list[StrPath]
            Search for matching configs in this list.

        Yields
        ------
        Generator[list[StrPath], Any, None]
            A list of macthing configs.
        """
        while configs:
            combined_config = [configs.pop()]
            current_file_id = combined_config[0].split(".")[2]
            for config in deepcopy(configs):
                if config.split(".")[2] == current_file_id:
                    combined_config.append(configs.pop())
            yield combined_config

    @classmethod
    def applyConfigs(cls, configs: list[StrPath]) -> None:
        for config in configs:
            file = os.path.split(config)[1]
            if config.find("gridparams") != -1:
                SetupConfig.grid_config_file = file
                SetupConfig.grid_config_path = config
            elif config.find("pipeline_config") != -1:
                SetupConfig.pipeline_config_file = file
                SetupConfig.pipeline_config_path = config
            else:
                cls._logger.error(
                    f"Unrecognized config '{config}'. It is most likely missing from '{SetupConfig.__name__}'"
                )
