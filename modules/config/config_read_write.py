import json
import os
import traceback

from pathlib import Path
from typing import Any, Literal, Optional

from modules.config.setup_config import SetupConfig
from modules.exceptions import MissingFieldError
from modules.logging import logger
from modules.types import StrPath

# REVIEW: why is this not wrapped in a class? - TODO:make class


def writeConfig(
    config: dict,
    destionation_path: StrPath,
) -> None:
    """
    Convert and write a Python config object to config files of different types.
    Supports: toml, ini, json.

    Parameters
    ----------
    config : dict | Model
        The Python config object to convert and write to a file.
        Can be an instance of a validation model.

    destionation_path : StrPath
        Path-like object pointing to a supported config file.
        Note: the file does not have to exist.
    """
    destination_directory = Path(os.path.dirname(destionation_path))
    # Split to get file name from path. E.g. "modules\config\setup_config.py", get "setup_config.py"
    file_name = os.path.split(destionation_path)[1]
    extension = os.path.splitext(destionation_path)[1].strip(".")
    try:
        destination_directory.mkdir(parents=True, exist_ok=True)

        if extension.lower() == "json":
            _generateJSONConfig(config, destionation_path)
        else:
            logger.warning(f"Cannot write unsupported file '{file_name}'")
    except Exception:
        # Catch exception to provide more descriptive cause of error, then re-raise exception to fail fast
        logger.error(
            f"Failed to write {file_name} to '{destionation_path}'\n"
            + traceback.format_exc(limit=SetupConfig.traceback_limit)
        )
        raise  # Re-raise the current exception


def _generateJSONConfig(config: dict, destination_path: StrPath) -> None:
    """
    Convert a Python config object to the '.json'-format and write it to a '.json' file.

    Parameters
    ----------
    config : dict
        A Python config object

    destination_path : StrPath
        Path-like object pointing to a JSON file.
        Note: the file does not have to exist.
    """
    file_name = os.path.split(destination_path)[1]
    with open(destination_path, "w", encoding="utf-8") as file:
        logger.debug(f"Writing '{file_name}' to '{destination_path}'")
        file.write(json.dumps(config, indent=4))


def checkMissingFields(config: dict, template: dict) -> None:
    """
    Compare the config against the template for missing
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
        """
        Helper function to keep track of parents while traversing.
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
            # We've reached the bottom level of the dict nesting (non-dict key/value pairs)
            elif key not in validation_dict:
                if parents:
                    field_errors.append(
                        f"{search_mode.capitalize()} setting '{key}' in {f"section '{parents[0]}'" if len(parents) == 1 else f"subsection '{".".join(parents)}'"}"
                    )
                else:
                    field_errors.append(f"{search_mode.capitalize()} setting '{key}'")
        else:
            if parents:
                # We're done with this dict level. Thus, its parent is no longer applicable
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
    """
    Return first value found.

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
        for stack_key, stack_value in stack[-1]:
            # The stack key is what we're looking for
            if stack_key == key:
                # The key need a parent to be considered relevant
                if parent_key:
                    # The stack key has the correct parent key
                    if parent_key in parent_keys:
                        found_value = stack_value
                        stack.clear()
                        break
                # The stack key does not need a parent
                else:
                    found_value = stack_value
                    stack.clear()
                    break
            # This level of the dict is still nested
            elif isinstance(stack_value, dict):
                stack.append(iter(stack_value.items()))
                parent_keys.append(stack_key)
                break
        else:
            stack.pop()  # We're reached the end of this dict level. Remove from stack
            if parent_keys:
                # We're done with this dict level. Thus, its parent is no longer applicable
                parent_keys.pop()
    try:
        return found_value
    # If we did not find a value, `found_value` is unbound (undefined)
    except UnboundLocalError as err:
        err.add_note(f"Error: Key '{key}' does not exists")
        raise  # Re-raise the current exception (UnboundLocalError) to fail fast


# TODO: Make dynamic programming version
def insertDictValue(
    input: dict, key: str, value: Any, parent_key: Optional[str] = None
) -> None:
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

    Raises
    ------
    KeyError
        If the key was not found in the config.
    """
    old_value = []  # Modified in-place by traverseDict
    parent_keys = []

    # Nested function is ONLY used in this function. Placement of source code considered more readable here.
    def traverseDict(
        _input: dict, search_key: str, value: Any, _parent_key: str
    ) -> None:
        """
        Recursively search through `_input` depth-first.

        Parameters
        ----------
        _input : dict
            Dict to search in.

        search_key : str
            The key to search for.

        value : Any
            The value assigned to `search_key`.

        _parent_key : str
            `search_key` must have this key as a parent.
        """
        for traverse_key, traverse_value in _input.items():
            # TODO: rewrite condition e.g. by raising an exception (remove old_value in favor of exceptions)
            # We've found the value we're looking for
            if old_value:
                break
            # TODO: Rewrite cases for clarity

            # The dict is still nested
            if isinstance(traverse_value, dict):
                parent_keys.append(traverse_key)
                traverseDict(traverse_value, search_key, value, _parent_key)

            # The key is what we're looking for
            if traverse_key == search_key:
                # The key need a parent to be considered relevant
                if parent_key:
                    # The key has the correct parent key
                    if _parent_key in parent_keys:
                        _input[traverse_key] = value
                        old_value.append(traverse_value)
                # The key does not need a parent
                else:
                    _input[traverse_key] = value
                    old_value.append(traverse_value)
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
    """
    Read and validate the config file residing at the supplied config path.

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
        If the file at the config path is not supported.
    """
    # TODO: Nesting level is too high - needs refactoring
    is_error = False
    config = None
    filename = os.path.split(config_path)[1]
    extension = os.path.splitext(filename)[1].strip(".")
    try:
        # Read file from disk
        with open(config_path, "rb") as file:
            if extension.lower() == "json":
                config = json.load(file)
            else:
                err_msg = f"{config_name}: Cannot load unsupported file '{config_path}'"
                raise NotImplementedError(err_msg)

        # Validate the read config (its considered valid if no exceptions occur)
        checkMissingFields(config, template)
    except MissingFieldError as err:
        is_error, is_recoverable = True, True
        err_msg = f"{config_name}: Detected incorrect fields in '{filename}':\n"
        # Format error message for printing
        for item in err.args[0]:
            err_msg += f"  {item}\n"
        logger.warning(err_msg)
        logger.info(f"{config_name}: Repairing config")
        repaired_config = repairConfig(config, template)
        writeConfig(repaired_config, config_path)
    except json.JSONDecodeError as err:
        is_error, is_recoverable = True, True
        logger.warning(f"{config_name}: Failed to parse '{filename}':\n  {err.msg}\n")
        writeConfig(template, config_path)
    except FileNotFoundError:
        is_error, is_recoverable = True, True
        logger.info(f"{config_name}: Creating '{filename}'")
        writeConfig(template, config_path)
    except Exception:
        is_error, is_recoverable = True, False
        logger.error(
            f"{config_name}: An unexpected error occurred while loading '{filename}'\n"
            + traceback.format_exc(limit=SetupConfig.traceback_limit)
        )
    finally:
        if is_error:
            # Reload the config if no fatal exceptions occured (i.e. only those we can recover from)
            if retries > 0 and is_recoverable:
                logger.info(f"{config_name}: Reloading '{filename}'")
                config = loadConfig(
                    config_name=config_name,
                    config_path=config_path,
                    template=template,
                    retries=retries - 1,
                )
            # We failed to load the config
            else:
                load_failure_msg = f"{config_name}: Failed to load '{filename}'"
                if template:
                    load_failure_msg += ". Switching to template config"
                    config = template  # Use template config if all else fails
                    logger.warning(load_failure_msg)
                else:
                    # We have no way of recovering
                    logger.error(load_failure_msg)
        else:
            logger.info(f"{config_name}: Config '{filename}' loaded!")
        return config


def repairConfig(config: dict, template: dict) -> dict:
    """
    Preserve all valid values in `config` when some of its fields are determined invalid.
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
        # Search config/template recursively, depth-first
        if isinstance(value, dict) and template_key in config:
            repaired_config |= {template_key: repairConfig(config[template_key], value)}
        # Preserve value from config
        elif template_key in config:
            repaired_config |= {template_key: config[template_key]}
        # Use value from template
        else:
            repaired_config |= {template_key: value}
    return repaired_config
