import json
import os

from modules.config.setup_config import SetupConfig
from modules.exceptions import MissingFieldError
from modules.logging import logger
import traceback

from pathlib import Path
from typing import Any, Literal, Optional, TypeAlias


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

        if extension.lower() == "json":
            _generateJSONConfig(config, dst_path)
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


def _generateJSONConfig(config: dict, dstPath: StrPath) -> None:
    """Convert a Python config object to the '.json'-format and write it to a '.json' file.

    Parameters
    ----------
    config : dict
        A Python config object

    dstPath : StrPath
        Path-like object pointing to a JSON file.
        Note: the file does not have to exist.
    """
    fileName = os.path.split(dstPath)[1]
    with open(dstPath, "w", encoding="utf-8") as file:
        logger.debug(f"Writing '{fileName}' to '{dstPath}'")
        file.write(json.dumps(config, indent=4))


def checkMissingFields(config: dict, template: dict) -> None:
    """Compare the config against the template for missing
    sections/settings and vice versa.

    Parameters
    ----------
    config : dict
        The config loaded from a file.

    template : dict
        The template used to create `config`.

    Raises
    ------
    MissingFieldError
        If any missing or unknown sections/settings are found.
    """
    all_errors, section_errors, field_errors = [], [], []
    parents = []

    def searchFields(
        config: dict, template: dict, search_mode: Literal["missing", "unknown"]
    ) -> None:
        """Helper function to keep track of parents while traversing.
        Use `search_mode` to select which type of field to search for.

        Parameters
        ----------
        config : dict
            The config loaded from a file.

        template : dict
            The template used to create `config`.

        search_mode : Literal["missing", "unknown"]
            Specify which type of field to search for.
        """
        # The dict to search depth-first in. Should be opposite of validation dict
        search_dict = template if search_mode == "missing" else config
        # The dict to compare the search dict against. Should be opposite of search dict
        validation_dict = config if search_mode == "missing" else template
        for key, value in search_dict.items():
            # The template is still nested (dict key/value pairs, i.e., sections)
            if isinstance(value, dict):
                if key in validation_dict:  # section exists
                    parents.append(key)
                    next_search = (
                        (config[key], value)
                        if search_mode == "missing"
                        else (value, template[key])
                    )
                    searchFields(*next_search, search_mode=search_mode)
                else:
                    section_errors.append(
                        f"{search_mode.capitalize()} {f"subsection '{".".join(parents)}.{key}'" if parents else f"section '{key}'"}"
                    )
            # We've reached the bottom of the nesting (non-dict key/value pairs)
            elif key not in validation_dict:
                if parents:
                    field_errors.append(
                        f"{search_mode.capitalize()} setting '{key}' in {f"section '{parents[0]}'" if len(parents) == 1 else f"subsection '{".".join(parents)}'"}"
                    )
                else:
                    field_errors.append(f"{search_mode.capitalize()} setting '{key}'")
        else:
            if parents:
                parents.pop()

    searchFields(config, template, search_mode="missing")
    searchFields(config, template, search_mode="unknown")

    # Ensure all section errors are displayed first
    all_errors.extend(section_errors)
    all_errors.extend(field_errors)
    if len(all_errors) > 0:
        raise MissingFieldError(all_errors)


# TODO: create dynamic programming version
def retrieveDictValue(
    input: dict,
    key: str,
    parent_key: Optional[str] = None,
) -> Any:
    """Return first value found.

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
    template: dict[str, Any],
    retries: int = 1,
) -> dict[str, Any] | None:
    """Read and validate the config file residing at the supplied config path.

    Parameters
    ----------
    config_name : str
        The name of the config

    config_path : StrPath
        Path-like object pointing to a config file.

    template : dict[str, Any]
        The template used to create the loaded config.

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
            if extension.lower() == "json":
                config = json.load(file)
            else:
                err_msg = f"{config_name}: Cannot load unsupported file '{config_path}'"
                raise NotImplementedError(err_msg)
        checkMissingFields(config, template)
    except MissingFieldError as err:
        isError, isRecoverable = True, True
        err_msg = f"{config_name}: Detected incorrect fields in '{filename}':\n"
        for item in err.args[0]:
            err_msg += f"  {item}\n"
        logger.warning(err_msg)
        logger.info(f"{config_name}: Repairing config")
        repairedConfig = repairConfig(config, template)
        writeConfig(repairedConfig, config_path)
    except json.JSONDecodeError as err:
        isError, isRecoverable = True, True
        logger.warning(f"{config_name}: Failed to parse '{filename}':\n  {err.msg}\n")
        writeConfig(template, config_path)
    except FileNotFoundError:
        isError, isRecoverable = True, True
        logger.info(f"{config_name}: Creating '{filename}'")
        writeConfig(template, config_path)
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
                    template=template,
                    retries=retries - 1,
                )
            else:
                load_failure_msg = f"{config_name}: Failed to load '{filename}'"
                if template:
                    load_failure_msg += ". Switching to template config"
                    config = template  # Use template config if all else fails
                    logger.warning(load_failure_msg)
                else:
                    logger.error(load_failure_msg)
        else:
            logger.info(f"{config_name}: Config '{filename}' loaded!")
        return config


def repairConfig(config: dict, template: dict) -> dict:
    """Preserve all valid values in `config` when some of its fields are determined invalid.
    Fields are taken from `template` if they could not be preserved from `config`.

    Parameters
    ----------
    config : dict
        The config loaded from a file.

    template : dict
        The template used to create `config`.

    Returns
    -------
    dict
        A new config where all values are valid with as many as possible
        preserved from `config`.
    """
    repaired_config = {}
    for template_key, value in template.items():
        if isinstance(value, dict) and template_key in config:
            # Search config/template recursively, depth-first
            repaired_config |= {template_key: repairConfig(config[template_key], value)}
        elif template_key in config:
            # Preserve value from config
            repaired_config |= {template_key: config[template_key]}
        else:
            # Use value from template
            repaired_config |= {template_key: value}
    return repaired_config
