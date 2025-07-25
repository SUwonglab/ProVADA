"""
log.py

This module contains the logging configuration for the codebase.
"""

import os
import logging
from provada.paths import REPO_ROOT
from typing import Optional
import sys
import threading
import coloredlogs
from IPython import get_ipython

def is_running_in_notebook():
    """
    Determines if the code is being executed in a Jupyter Notebook or IPython environment.
    """
    try:
        # The get_ipython function is available in IPython environments (including Jupyter).
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type of shell (e.g., IDLE)
    except NameError:
        if "ipykernel" in sys.modules:
            return True

    return False  # Probably standard Python interpreter

_is_configured = False
_lock = threading.Lock()

VALID_PREFIXES = ["provada", "run_provada"]

# The format of the log messages in the file
FILE_FORMAT = "%(levelname)s:%(asctime)s:%(name)s:%(message)s"
DATE_FORMAT = "%H-%M-%S"


# Format of log messages in the console (used by coloredlogs)
CONSOLE_FORMAT = "%(levelname)s: %(message)s"

# This class is now only used if you are in a notebook
class NotebookFormatter(logging.Formatter):
    def format(self, record):
        # Add level name prefix only for WARNING and above
        if record.levelno >= logging.WARNING:
            record.msg = f"{record.levelname}: {record.msg}"
        # For DEBUG and INFO, leave msg unchanged
        return super().format(record)


# The absolute path to the log directory
LOG_DIR = os.path.join(str(REPO_ROOT), "logs")


class CodebaseFilter(logging.Filter):
    def filter(self, record):
        return any(prefix in record.name for prefix in VALID_PREFIXES)


def setup_logger(
    verbose: bool = False,
    log_filename: Optional[str] = None,
    logging_subdir: Optional[str] = None,
):
    """
    Configures the root logger for the codebase

    Args:
        verbose (bool): If True, the stream handler will be set to DEBUG level.
        log_filename (Optional[str]): The name of the log file. If None, no file
            logging will be done.
        logging_subdir (Optional[str]): The name of the subdirectory to log to.
            If None, the log file will be in the root log directory.
    """
    global _is_configured

    # Get the root logger
    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set the root logger level to DEBUG by default
    root_logger.setLevel(logging.DEBUG)

    # --- Start of coloredlogs integration ---

    # If running in a notebook, use the original, simpler formatter.
    # Otherwise, install coloredlogs for a rich terminal experience.
    if is_running_in_notebook():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(NotebookFormatter())
        ch.addFilter(CodebaseFilter())
        root_logger.addHandler(ch)
    else:
        # Install coloredlogs, which will handle the console logging.
        coloredlogs.install(
            level=logging.DEBUG if verbose else logging.INFO,
            fmt=CONSOLE_FORMAT,
            stream=sys.stdout,
            # Pass your custom filter to coloredlogs
            custom_filters=[CodebaseFilter()],
        )

    # --- End of coloredlogs integration ---

    # Add file handler (if file logging is enabled) - this part remains the same
    if log_filename is not None:

        if logging_subdir is None and "WANDB_SWEEP_ID" in os.environ:
            logging_subdir = os.path.join(os.environ["WANDB_SWEEP_ID"], "run_logs")

        # Determine the log file path
        log_filepath = None
        if logging_subdir is not None:
            log_filepath = os.path.join(LOG_DIR, logging_subdir, log_filename)
        else:
            log_filepath = os.path.join(LOG_DIR, log_filename)

        if not log_filepath.endswith(".log"):
            log_filepath += ".log"

        # Create the log file directory if it doesn't exist
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

        # Create the file handler
        fh = logging.FileHandler(filename=log_filepath, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        fh.addFilter(CodebaseFilter())
        root_logger.addHandler(fh)

    root_logger.info(f"Logging configured")
    if log_filename is not None:
        root_logger.info(f"Logging to file: {log_filepath}")
        root_logger.log_filepath = log_filepath

    # Set the global flag to True
    _is_configured = True


def get_logger(
    name: str,
) -> logging.Logger:
    global _is_configured

    with _lock:
        if not _is_configured:
            setup_logger()

    return logging.getLogger(name)


import argparse
import logging
from typing import Union, List, Any

# Set up a logger, as in the original code
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Use a try-except block for OmegaConf imports to make the code portable
# It will function correctly even if OmegaConf is not installed.
try:
    from omegaconf import DictConfig, ListConfig, OmegaConf

    # Define types for broader compatibility
    DICT_LIKE = (DictConfig, dict, argparse.Namespace)
    LIST_LIKE = (ListConfig, list)
    ANY_CONFIG = Union[DictConfig, ListConfig, dict, list, argparse.Namespace]
except ImportError:
    # Fallback definitions if OmegaConf is not installed
    DictConfig = ListConfig = OmegaConf = None
    DICT_LIKE = (dict, argparse.Namespace)
    LIST_LIKE = (list,)
    ANY_CONFIG = Union[dict, list, argparse.Namespace]


def _generate_tree_lines(config: ANY_CONFIG, prefix: str = "") -> List[str]:
    """
    Recursively builds the list of strings for the configuration tree.

    This helper function handles the core traversal logic for all supported types.
    """
    lines = []

    # --- Handle Dict-like objects (dict, DictConfig, Namespace) ---
    if isinstance(config, DICT_LIKE):
        # Convert Namespace to dict for uniform processing
        config_dict = vars(config) if isinstance(config, argparse.Namespace) else config

        items = list(config_dict.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "|   ")

            if isinstance(value, (DICT_LIKE, LIST_LIKE)):
                lines.append(f"{prefix}{connector}{key}:")
                lines.extend(_generate_tree_lines(value, child_prefix))
            else:
                lines.append(f"{prefix}{connector}{key}: {value}")
        return lines

    # --- Handle List-like objects (list, ListConfig) ---
    elif isinstance(config, LIST_LIKE):
        for i, item in enumerate(config):
            is_last = i == len(config) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "|   ")

            if isinstance(item, (DICT_LIKE, LIST_LIKE)):
                # Use a hyphen for complex items (dicts/lists) inside a list
                lines.append(f"{prefix}{connector}-")
                lines.extend(_generate_tree_lines(item, child_prefix))
            else:
                lines.append(f"{prefix}{connector}{item}")
        return lines

    return lines


def display_config(
    config: ANY_CONFIG, config_name: str = "Config", resolve: bool = True
) -> str:
    """
    Recursively prints and logs the content of a configuration object as a tree.
    Supports argparse.Namespace, OmegaConf types, and standard Python dicts/lists.

    Args:
        config (ANY_CONFIG): The configuration object to display.
        config_name (str, optional): The root name for the tree. Defaults to "Config".
        resolve (bool, optional): For OmegaConf, whether to resolve interpolations
                                (e.g., `${...}`). Defaults to True.

    Returns:
        str: The formatted configuration as a string.
    """
    processed_config = config

    # If OmegaConf is available and the input is an OmegaConf object,
    # convert it to a standard Python container, respecting the 'resolve' flag.
    if OmegaConf and isinstance(config, (DictConfig, ListConfig)):
        processed_config = OmegaConf.to_container(config, resolve=resolve)

    # Start the tree with the root name
    lines = [f"{config_name}:"]
    lines.extend(_generate_tree_lines(processed_config))

    config_text = "\n".join(lines)
    logger.info(config_text)
