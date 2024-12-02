import hashlib
import os
import shutil
from pathlib import Path
from datetime import datetime

from modules.config.config import Config
from modules.config.grid_config import GridConfig
from modules.config.utils.config_batch_processor import ConfigBatchProcessor
from modules.config.utils.setup_config import SetupConfig
from modules.logging import logger
from modules import tools
from modules.tools.types import StrPath


class ConfigExporter:
    _logger = logger

    def __init__(self, bufsize: int = 65536) -> None:
        """
        Convenience class for exporting the currently active configs to disk
        and checking for duplicate configs before exporting.

        Parameters
        ----------
        bufsize : int, optional
            Read file buffer size in bytes.
            By default `65536`.
        """
        self._bufsize = bufsize

    def _hashFileContent(self, path: StrPath) -> str:
        """
        Reads a file and hashes its content using the SHA1 algorithm.

        Parameters
        ----------
        path : StrPath
            Path to the file.

        Returns
        -------
        str
            SHA1 hash of the file's content
        """
        sha1 = hashlib.sha1()

        with open(path, "rb") as file:
            while True:
                data = file.read(self._bufsize)
                if not data:
                    # We've reached EOF
                    break
                sha1.update(data)
        return sha1.hexdigest()

    def _compareFileHashes(self, file_paths: list[StrPath]) -> bool:
        """
        Compares hash of file content for all files in `file_paths` to ensure each file is unique.

        Parameters
        ----------
        file_paths : list[StrPath]
            Paths of files which to compare.

        Returns
        -------
        bool
            True if files in `file_paths` are unique.
            False otherwise.
        """
        is_unique = True
        disk_files = {}
        for file_path in file_paths:
            file_hash = self._hashFileContent(file_path)
            if file_hash in disk_files.values():
                self._logger.warning(
                    f"Duplicate files detected! File '{file_path}' is a duplicate of '{tools.dictLookup(disk_files, file_hash)}'"
                )
                is_unique = False
            disk_files[file_path] = file_hash
        return is_unique

    def exportConfigs(self) -> None:
        """Export the active configs to disk with a unique name and check for duplicate configs."""
        # Create export directory
        os.makedirs(SetupConfig.arg_export_path, exist_ok=True)

        # The currently active configs
        config_paths = [Config()._config_path, GridConfig()._config_path]

        # Check that the configs-to-be-exported are unique compared to existing configs on disk
        file_paths = [
            *config_paths,
            *ConfigBatchProcessor.getBatchConfigs(SetupConfig.arg_export_path),
        ]
        self._compareFileHashes(file_paths)

        # Export the current configs by creating a copy of each with a unique name
        # The system time is used as a unique ID for this collection of configs.
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for config_path in config_paths:
            file = os.path.split(config_path)[1]
            file_name, extension = os.path.splitext(file)
            file_name = f"exported.{file_name}.{time}{extension}"
            shutil.copyfile(config_path, Path(SetupConfig.arg_export_path, file_name))

        self._logger.info(f"All configs exported succesfully!")
