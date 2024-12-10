from copy import deepcopy
import glob
import os
from typing import Any, Generator
from modules.config.grid_config import GridConfig
from modules.config.pipeline_config import PipelineConfig
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
            A list of matching configs.
        """
        while configs:
            combined_config = [configs.pop()]
            current_filename = os.path.split(combined_config[0])[1]
            # Dirty fix
            try:
                current_file_id = current_filename.split(".")[2]
            except IndexError:
                current_file_id = current_filename

            for config in deepcopy(configs):
                new_filename = os.path.split(config)[1]  # Filename with extension
                # Dirty fix
                try:
                    new_file_id = new_filename.split(".")[2]
                except IndexError:
                    new_file_id = new_filename

                if new_file_id == current_file_id:
                    combined_config.append(configs.pop())
            yield combined_config

    @classmethod
    def applyConfigs(cls, configs: list[StrPath]) -> None:
        for config in configs:
            file = os.path.split(config)[1]
            if config.find("gridparams") != -1:
                SetupConfig.grid_config_file = file
                SetupConfig.grid_config_path = config
                setattr(GridConfig, "_created", False)
            elif config.find("pipeline_config") != -1:
                SetupConfig.pipeline_config_file = file
                SetupConfig.pipeline_config_path = config
                pipconf = PipelineConfig()
                # Reset config such that a new instance is created
                pipconf._created = False
                pipconf._instance = None
            else:
                cls._logger.warning(
                    f"Unrecognized config '{config}'. Assuming it is 'pipeline_config'"
                )
                SetupConfig.pipeline_config_file = file
                SetupConfig.pipeline_config_path = config
                pipconf = PipelineConfig()
                # Reset config such that a new instance is created
                pipconf._created = False
                pipconf._instance = None
