import os
from modules.config.setup_config import SetupConfig
from modules.exceptions import MissingFieldError
from modules.logging import logger
import tomlkit
import tomlkit.exceptions
import traceback

from pathlib import Path
from typing import Any, Optional, TypeAlias


# REVIEW: why is this not wrapped in a class? - TODO:make class
StrPath: TypeAlias = str | os.PathLike[str]


def writeConfig(
    config: dict,
    dst_path: StrPath,
) -> None:
    """
    Convert and write a Python config object to config files of different types.
    Supports: toml, ini, json.

    Parameters
    ----------
    config : dict | Model
        The Python config object to convert and write to a file.
        Can be an instance of a validation model.

    dst_path : StrPath
        Path-like object pointing to a supported config file.
        Note: the file does not have to exist.
    """
    dst_dir = Path(os.path.dirname(dst_path))
    file = os.path.split(dst_path)[1]
    extension = os.path.splitext(dst_path)[1].strip(".")
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)

        if extension.lower() == "toml":
            _generateTOMLconfig(config, dst_path)
        else:
            logger.warning(f"Cannot write unsupported file '{file}'")
    # We don't know which exceptions would be thrown, we just want some logging
    except Exception:
        logger.error(
            f"Failed to write {file} to '{dst_path}'\n"
            + traceback.format_exc(limit=SetupConfig.traceback_limit)
        )
        # Reraise the current exception
        raise


def _generateTOMLconfig(config: dict, dstPath: StrPath) -> None:
    """Convert a Python config object to the '.toml'-format and write it to a '.toml' file.

    Parameters
    ----------
    config : dict
        A Python config object.

    dstPath : StrPath
        Path-like object pointing to a toml file.
        Note: the file does not have to exist.
    """
    fileName = os.path.split(dstPath)[1]
    doc = tomlkit.document()
    for section, section_table in config.items():
        # NOTE: Toml table corresponds to key/value pairs in a dict
        table = tomlkit.table()
        for key in section_table:
            value = section_table[key]
            table.append(key, value)
        doc.append(section, table)

    # NOTE: exception caught by caller (writeConfig)
    with open(dstPath, "w", encoding="utf-8") as file:
        logger.debug(f"Writing '{fileName}' to '{dstPath}'")
        tomlkit.dump(doc, file)


def checkMissingFields(disk_config: dict, template_config: dict) -> None:
    """Compare the raw_config against the template_config for missing
    sections/settings and vice versa.

    Parameters
    ----------
    raw_config : dict
        A config read from a file.

    template_config : dict
        The template version of the raw config file.

    Raises
    ------
    MissingFieldError
        If any missing or unknown sections/settings are found.
    """
    # TODO: Find more optimal way to compare files and check validity
    allErrors, sectionErrors, fieldErrors = [], [], []
    for section in template_config:  # Check sections
        sectionExistsInConfig = section in disk_config
        if not sectionExistsInConfig:
            sectionErrors.append(f"Missing section '{section}'")
        else:
            for setting in template_config[section]:  # Check settings in a section
                if sectionExistsInConfig and setting not in disk_config[section]:
                    fieldErrors.append(
                        f"Missing setting '{setting}' in section '{section}'"
                    )

    for section in disk_config:  # Check sections
        sectionShouldExist = section in template_config
        if not sectionShouldExist:
            if isinstance(disk_config[section], dict):
                sectionErrors.append(f"Unknown section '{section}'")
            else:
                fieldErrors.append(f"Setting '{section}' does not belong to a section")
        else:
            for setting in disk_config[section]:  # Check settings in a section
                if sectionShouldExist and setting not in template_config[section]:
                    fieldErrors.append(
                        f"Unknown setting '{setting}' in section '{section}'"
                    )
    # Ensure all section errors are displayed first
    allErrors.extend(sectionErrors)
    allErrors.extend(fieldErrors)
    if len(allErrors) > 0:
        raise MissingFieldError(allErrors)


# TODO: create dynamic programming version
def retrieveDictValue(
    input: dict,
    key: str,
    parent_key: Optional[str] = None,
) -> Any:
    """Return first value found.
    If `key` does not exists, return `default`.

    Has support for defining search scope with `parent_key`;
    A value will only be returned if it is within the scope of `parent_key`.

    Parameters
    ----------
    d : dict
        The dictionary to search for `key`.

    key : str
        The key to search for.

    parent_key : str, optional
        Limit the search scope to the children of `key`.

    Returns
    -------
    Any
        The value mapped to `key` if it exists.

    Raises
    ------
    UnboundLocalError
        If `key` was not found in the dict.

    """
    stack = [iter(input.items())]
    parent_keys = []
    while stack:
        for k, v in stack[-1]:
            if k == key:
                if parent_key:
                    if parent_key in parent_keys:
                        found_value = v
                        stack.clear()
                        break
                else:
                    found_value = v
                    stack.clear()
                    break
            elif isinstance(v, dict):
                stack.append(iter(v.items()))
                parent_keys.append(k)
                break
        else:
            stack.pop()
            if parent_keys:
                parent_keys.pop()
    try:
        return found_value
    except UnboundLocalError as err:
        err.add_note(f"Error: Key {key} does not exists")
        raise


# TODO: Make dynamic programming version
def insertDictValue(
    input: dict, key: str, value: Any, parent_key: Optional[str] = None
) -> list | None:
    """
    Recursively look for key in input.
    If found, replace the original value with the provided value and return the original value.

    Has support for defining search scope with the parent key.
    Value will only be returned if it is within parent key's scope.

    Note: If a nested dict with multiple identical parent_keys exist,
    only the top-most parent_key is considered

    Causes side-effects!
    ----------
    Modifies input in-place (i.e. does not return input).

    Parameters
    ----------
    input : dict
        The dictionary to search in.

    key : str
        The key to look for.

    value : Any
        The value to insert.

    parent_key : str, optional
        Limit the search scope to the children of this key.
        By default None.

    Returns
    -------
    list | None
        The replaced old value, if found. Otherwise, None.

    Raises
    ------
    KeyError
        If the key was not found in the config.
    """
    old_value = []  # Modified in-place by traverseDict
    parent_keys = []

    # Nested function is ONLY used in this function. Placement of source code considered more readable here.
    def traverseDict(_input: dict, _key, _value, _parent_key) -> list | None:
        for k, v in _input.items():
            # TODO: rewrite condition e.g. by raising an exception (remove old_value in favor of exceptions)
            if old_value:
                break
            # TODO: Rewrite cases for clarity
            if isinstance(v, dict):
                parent_keys.append(k)
                traverseDict(v, _key, _value, _parent_key)
            elif k == _key:
                if parent_key:
                    if _parent_key in parent_keys:
                        _input[k] = _value
                        old_value.append(v)
                else:
                    _input[k] = _value
                    old_value.clear()
                    old_value.append(v)
                break

    traverseDict(input, key, value, parent_key)
    if not old_value:
        raise KeyError(f"Error: Key {key} does not exists")


def loadConfig(
    config_name: str,
    config_path: StrPath,
    template_config: Optional[dict[str, Any]] = None,
    retries: int = 1,
) -> dict[str, Any] | None:
    """Read and validate the config file residing at the supplied config path.

    Parameters
    ----------
    config_name : str
        The name of the config

    config_path : StrPath
        Path-like object pointing to a config file.

    template_config : dict[str, Any] | None, optional
        The template connected to the loaded config
        By default None.

    retries : int, optional
        Reload the config X times if soft errors occur.
        By default 1.

    Returns
    -------
    dict[str, Any] | None
        The config file converted to a dict or None if loading failed.

    Raises
    ------
    NotImplementedError
        If the file at the config path is not supported
    """
    # TODO: Nesting level is too high - needs refactoring
    isError = False
    config = None
    filename = os.path.split(config_path)[1]
    extension = os.path.splitext(filename)[1].strip(".")
    try:
        with open(config_path, "rb") as file:
            if extension.lower() == "toml":
                raw_config = tomlkit.load(file)
            else:
                err_msg = f"{config_name}: Cannot load unsupported file '{config_path}'"
                raise NotImplementedError(err_msg)
        checkMissingFields(raw_config, template_config)
        config = raw_config
    except MissingFieldError as err:
        isError, isRecoverable = True, True
        err_msg = f"{config_name}: Detected incorrect fields in '{filename}':\n"
        for item in err.args[0]:
            err_msg += f"  {item}\n"
        logger.warning(err_msg)
        logger.info(f"{config_name}: Repairing config")
        repairedConfig = repairConfig(raw_config, template_config)
        writeConfig(repairedConfig, config_path)
    except tomlkit.exceptions.ParseError as err:
        isError, isRecoverable = True, True
        logger.warning(
            f"{config_name}: Failed to parse '{filename}':\n  {err.args[0]}\n"
        )
        writeConfig(template_config, config_path)
    except FileNotFoundError:
        isError, isRecoverable = True, True
        logger.info(f"{config_name}: Creating '{filename}'")
        writeConfig(template_config, config_path)
    except Exception:
        isError, isRecoverable = True, False
        logger.error(
            f"{config_name}: An unexpected error occurred while loading '{filename}'\n"
            + traceback.format_exc(limit=SetupConfig.traceback_limit)
        )
    finally:
        if isError:
            if retries > 0 and isRecoverable:
                logger.info(f"{config_name}: Reloading '{filename}'")
                config = loadConfig(
                    config_name=config_name,
                    config_path=config_path,
                    template_config=template_config,
                    retries=retries - 1,
                )
            else:
                load_failure_msg = f"{config_name}: Failed to load '{filename}'"
                if template_config:
                    load_failure_msg += ". Switching to template config"
                    config = template_config  # Use template config if all else fails
                    logger.warning(load_failure_msg)
                else:
                    logger.error(load_failure_msg)
        else:
            logger.info(f"{config_name}: Config '{filename}' loaded!")
        return config


# TODO: write more thorough docstring esp. summary, what is the purpose of the function
def repairConfig(raw_dict: dict, template_config: dict) -> dict:
    """Preserve all valid values in the config when some fields are determined invalid.
    Note: does not support sectionless configs.

    Parameters
    ----------
    validated_config : dict
        The config, loaded from disk and validated, which contains invalid fields.

    template_config : dict
        The template of the `validated_config`.

    Returns
    -------
    dict
        The config where all values are valid with as many as possible
        preserved from the `validated_config`.
    """
    newConfig = {}
    for section_name, section in template_config.items():
        newConfig |= {section_name: {}}
        for setting, options in section.items():
            if section_name in raw_dict and setting in raw_dict[section_name]:
                newConfig[section_name] |= {setting: raw_dict[section_name][setting]}
            else:
                newConfig[section_name] |= {setting: options}
    return newConfig
