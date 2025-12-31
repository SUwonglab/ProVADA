"""
test_config.py
"""

import sys
import pytest
from typing import Iterable
from provada.utils.setup import get_config_from_args
from argparse import ArgumentParser


@pytest.fixture
def blank_argv(request, monkeypatch):
    """
    Set sys.argv to ['pytest', *extra].
    - By default: ['pytest'] so argparse sees no extra args.
    - If the test parametrizes this fixture, the provided items are appended.
      Example:
        @pytest.mark.parametrize("blank_argv", [["dataset.dataset_name=new_name"]], indirect=True)
        def test_...(): ...
    """
    extra = getattr(request, "param", None)

    if extra is None:
        extra_list = []
    elif isinstance(extra, str):
        # Allow passing a single string directly
        extra_list = [extra]
    elif isinstance(extra, Iterable):
        extra_list = list(extra)
    else:
        raise TypeError(
            f"blank_argv expects a string or iterable of strings, got {type(extra)}"
        )

    # Ensure argparse has a program name at index 0
    monkeypatch.setattr(sys, "argv", ["pytest", *extra_list])
    yield


@pytest.mark.usefixtures("blank_argv")
def test_only_config_loads_correctly():
    # Create a mock parser with only the config entry
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="mock_config.yaml")

    # Get the config
    config = get_config_from_args(parser)

    # Check that the config is loaded correctly
    assert config is not None, "Config is None"
    assert config.dataset.dataset_name == "gfp", "Mock config not loaded correctly"


@pytest.mark.usefixtures("blank_argv")
def test_known_args_override_config():
    # Create a mock parser with only the config entry
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="mock_config.yaml")

    # Add a known arg that is in the config
    parser.add_argument("--dataset_name", type=str, default="new_name")

    # Get the config
    config = get_config_from_args(parser)

    # Check that the config is loaded correctly
    assert config is not None, "Config is None"
    assert (
        config.dataset.dataset_name == "new_name"
    ), "Config attribute not overriden by known arg"


@pytest.mark.parametrize(
    "blank_argv",
    [["--dataset_name=new_name"]],
    indirect=True,
)
def test_unknown_args_accepted_when_in_config(blank_argv):
    """
    Tests to ensure that when arguments are in the config, they are accepted
    """

    # Create a mock parser with only the config entry
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="mock_config.yaml")

    # Get the config
    config = get_config_from_args(parser)

    # Check that the config is loaded correctly
    assert config is not None, "Config is None"
    assert (
        config.dataset.dataset_name == "new_name"
    ), "Config attribute not overriden by known arg"


@pytest.mark.parametrize(
    "blank_argv",
    [["--not_in_config=bad_arg"]],
    indirect=True,
)
def test_unknown_args_not_accepted_when_not_in_config(blank_argv):
    """
    Tests to ensure that when arguments are not in the config, they are not accepted
    """

    # Create a mock parser with only the config entry
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="mock_config.yaml")

    # Get the config
    try:
        get_config_from_args(parser)
    except ValueError as e:
        assert "Key not_in_config not found in config." in str(
            e
        ), "ValueError not raised when argument not in config"
