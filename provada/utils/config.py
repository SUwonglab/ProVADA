"""
configs.py

Utility functions for working with yaml based configuration files.
"""

import os
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Dict, Union
from argparse import Namespace, ArgumentParser
from provada.utils.files import get_filepath_from_name_or_path
from provada.utils.log import get_logger
from provada.utils.misc import set_nested_attr

logger = get_logger(__name__)


def load_config(config_name_or_path: str) -> DictConfig:
    """
    Load a configuration file by name or path. The config must be within the root
    directory of the project to be found (unless specified by abs path) and must
    have a .yaml or .yml extension.

    Args:
        config_name_or_path (str): The name or path of the configuration file to load.

    Returns:
        DictConfig: The loaded configuration file.
    """

    if os.path.exists(config_name_or_path) and os.path.isfile(config_name_or_path):
        logger.debug(f"Loading configuration file directly from {config_name_or_path}")
        # Return the config file directly from the path
        loaded_config = OmegaConf.load(config_name_or_path)

        file_name = Path(config_name_or_path).stem

    else:
        # Otherwise, we must find the config file
        config_filepath = get_filepath_from_name_or_path(
            config_name_or_path, [".yaml", ".yml"]
        )
        file_name = config_filepath["filename"]

        loaded_config = OmegaConf.load(config_filepath["absolute_path"])

    loaded_config.config_name = file_name

    return loaded_config


def update_config_from_args(
    original_config: DictConfig,
    args_to_update: Union[Namespace, Dict, DictConfig],
    require_keys_to_be_in_config: bool = False,
) -> DictConfig:
    """
    In place update of a configuration by matching keys specified in the namespace.
    Updates a configuration by matching keys specified in the namespace. Keys
    must be unique within the config to be updated.

    Args:
        original_config (DictConfig): The original configuration to update.
        args_to_update (Namespace): The namespace of arguments to update the config with.
        require_keys_to_be_in_config (bool, optional): Enforces that all keys in the args_to_update
            must be in the original_config.
    """

    # If the args_to_update is a DictConfig, we need to convert it to a dict
    if isinstance(args_to_update, DictConfig):
        args_to_update = OmegaConf.to_container(args_to_update, resolve=True)

    # Now, if the args_to_update is a dict, we need to convert it to a namespace
    if isinstance(args_to_update, dict):
        args_to_update = Namespace(**args_to_update)

    # First, get the config map of final keys
    config_map = get_config_map(original_config)

    # We also need to create a full key path map in case we have full key paths instead
    full_key_path_map = {}
    for short_key, map_list in config_map.items():
        for key_dict in map_list:
            full_key_path_map[key_dict["full_key_path"]] = key_dict | {"short_key": short_key}

    # Then, update the config with the values from the args
    for arg_key, arg_value in args_to_update.__dict__.items():

        # We only update values if they are not None
        if arg_value is not None:
            # Determine if the arg key is in the config map
            if arg_key in config_map:

                # Ensure the arg_key has a unique value
                if len(config_map[arg_key]) == 1:
                    set_nested_attr(
                        original_config,
                        config_map[arg_key][0]["full_key_path"],
                        arg_value,
                    )

                # Else, there are multiple values that have this first level key
                else:
                    conflict_list = f"Conflicts: {', '.join([b['full_key_path'] for b in config_map[arg_key]])}"
                    raise ValueError(
                        f"Multiple values found for {arg_key}.",
                        "Argument keys must be unique within the config to be updated.",
                        conflict_list,
                    )

            # Otherwise, if the argument is a full key path argument, just add it!
            elif arg_key in full_key_path_map:
                set_nested_attr(
                    original_config,
                    arg_key,
                    arg_value,
                )

            else:
                if require_keys_to_be_in_config:
                    raise ValueError(
                        f"Key {arg_key} not found in config.",
                        "All keys must be specified in the config to be updated.",
                    )
                else:
                    # If the key is not in the config map, we add it to the first level of the config
                    set_nested_attr(original_config, arg_key, arg_value)


def _map_config_helper(config, map_dict=None, key_path=None):
    """
    Helper function for mapping keys and values in the config.

    Args:
        config (OmegaConf): The configuration to map.
        map_dict (Dict, optional): The dictionary to store the mapped keys and values.
        key_path (List[str], optional): The path to the current key in the config.

    Returns:
        Dict: A dictionary with the final keys of all config values as the keys and
        the values as lists of dictionaries with the full key path, key path, and value.
    """
    if map_dict is None:
        map_dict = {}
    if key_path is None:
        key_path = []

    for k, v in config.items():
        if isinstance(v, (dict, OmegaConf, DictConfig)):
            _map_config_helper(v, map_dict, key_path + [k])
        else:
            full_key_path = ".".join(key_path + [k])
            if k not in map_dict:
                map_dict[k] = []

            map_dict[k].append(
                {"full_key_path": full_key_path, "key_path": key_path, "value": v}
            )
    return map_dict


def get_config_map(config):
    """
    Returns a map of the config based on the final keys of all config values.
    Helpful for updating configs from arg Namespace objects.

    Args:
        config (OmegaConf): The configuration to map.

    Returns:
        Dict: A dictionary with the final keys of all config values as the keys and
        the values as lists of dictionaries with the full key path, key path, and value.
    """
    return _map_config_helper(config)


def display_config(config, config_name="Config", resolve=True):
    """
    Recursively prints and logs the content of an OmegaConf configuration as plain text with a tree structure.

    Args:
        config (OmegaConf): The configuration to display.
        resolve (bool, optional): Whether to resolve reference fields in the DictConfig.

    Returns:
        str: The formatted configuration as a string.
    """
    lines = dict_config_str_builder(config, resolve, indent=0, prefix="")

    # Insert the config name at the beginning of the lines
    lines.insert(0, f"{config_name}:")

    config_text = "\n".join(lines)
    logger.info(config_text)

    return config_text


def flatten_config(config: DictConfig):
    """
    Flattens a config into a dictionary.
    """
    return _flatten_config_helper(config, {}, "")


def _flatten_config_helper(config: dict, flattened: dict, key: str):
    """
    Recursively flattens a config into a dictionary.
    """

    # Convert to dict if type of config is OmegaConf
    if (
        isinstance(config, OmegaConf)
        or isinstance(config, ListConfig)
        or isinstance(config, DictConfig)
    ):
        config = OmegaConf.to_container(config)

    for k, v in config.items():
        if type(v) is dict:
            _flatten_config_helper(v, flattened, f"{key}{k}.")
        elif type(v) is list:
            for ix, _config in enumerate(v):
                if type(_config) is dict:
                    _flatten_config_helper(_config, flattened, f"{key}{k}.{ix}_")
        else:
            flattened[f"{key}{k}"] = v
    return flattened


# ================================
# Config Display Helpers
# ================================


def dict_config_str_builder(config, resolve=True, indent=0, prefix=""):
    """
    Recursively builds the content of an OmegaConf configuration as plain text with a tree structure.

    This function is used to build the content of a dictionary within a config.

    Args:
        config (OmegaConf): The configuration to display.
        resolve (bool, optional): Whether to resolve reference fields in the DictConfig.
        indent (int, optional): Current indentation level, used for recursive formatting.
        prefix (str, optional): Prefix string used to create branch lines in the tree structure.
    """
    lines = []

    # Determine the proper continuation prefix based on parent
    if prefix.endswith("    "):
        prefix = prefix[:-4] + "    "
    else:
        prefix += "|   "

    is_dict = isinstance(config, DictConfig) or isinstance(config, dict)
    is_list = isinstance(config, ListConfig) or isinstance(config, list)

    if is_dict:
        keys = list(config.keys())
        last_key = keys[-1] if keys else None
        for key in keys:
            value = config[key]
            connector = "└── " if key == last_key else "├── "
            sub_prefix = prefix + ("    " if key == last_key else "|   ")
            if isinstance(value, (DictConfig, dict)):
                lines.append(f"{prefix}{connector}{key}:")
                lines.extend(dict_config_str_builder(value, resolve, indent + 1, sub_prefix))
            elif isinstance(value, (ListConfig, list)):
                lines.append(f"{prefix}{connector}{key}:")
                lines.extend(list_config_str_builder(value, resolve, sub_prefix))
            else:
                lines.append(f"{prefix}{connector}{key}: {value}")
    elif is_list:
        lines.extend(list_config_str_builder(config, resolve, prefix))

    return lines


def list_config_str_builder(lst, resolve, prefix):
    """
    Recursively builds the content of an OmegaConf configuration as plain text with a tree structure.

    This function is used to build the content of a list within a config.

    Args:
        lst (ListConfig): The list to build the content of.
        resolve (bool, optional): Whether to resolve reference fields in the DictConfig.
        prefix (str, optional): The prefix to use for the list.
    """
    last_index = len(lst) - 1
    for i, item in enumerate(lst):
        connector = "└── " if i == last_index else "├── "
        if isinstance(item, (DictConfig, dict, ListConfig, list)):
            if isinstance(item, (DictConfig, dict)):
                lines = dict_config_str_builder(
                    item, resolve, indent=0, prefix=prefix + connector
                )
            elif isinstance(item, list):
                lines = list_config_str_builder(item, resolve, prefix + connector)
            for line in lines:
                yield f"{prefix}{line}"
        else:
            yield f"{prefix}{connector}{item}"


def get_config_from_default_parser_args(parser: ArgumentParser):
    """
    Returns a config representations of arguments with non-None default values
    in a given parser. Useful for creating configs from a unutilized parser when
    running sweeps.
    """
    config = {}

    for action in parser._actions:
        if action.dest != "help" and action.default is not None:
            config[action.dest] = action.default

    # Return config
    return OmegaConf.create(config)
