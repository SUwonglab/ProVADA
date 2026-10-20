"""
env.py

Contains utility functions related to the environment
"""

import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from typing import Optional

from provada.utils.log import get_logger

logger = get_logger(__name__)


def resolve_device(device: Optional[str] = None) -> str:
    """
    Returns the device string to send to a GPU-backed proto-tools tool.

    proto-tools' tool configs default to ``"cpu"``, and a tool that declares no
    iterable input is never fanned out across a ``ToolPool``'s workers, so
    nothing overrides that default. Leaving the device unset therefore runs the
    model on CPU rather than on the GPUs the pool holds.

    This asks proto-tools what CUDA can see, so ``CUDA_VISIBLE_DEVICES`` remains
    the only thing that chooses placement: unset uses the node's GPUs, a subset
    pins, and the empty string gives CPU.

    Args:
        device: An explicit device to use instead, for a caller that wants one.

    Returns:
        ``device`` if given, else ``"cuda"`` when any GPU is visible and
        ``"cpu"`` when none is.
    """
    if device is not None:
        return device

    from proto_tools.utils.device import number_of_visible_gpus

    return "cuda" if number_of_visible_gpus() > 0 else "cpu"


@contextmanager
def suppress_console_output(stdout=True, stderr=True):
    """
    Suppress stdout and/or stderr output.

    Args:
        stdout (bool): Whether to suppress stdout
        stderr (bool): Whether to suppress stderr
    """
    stdout_redirect = redirect_stdout(StringIO()) if stdout else redirect_stdout(sys.stdout)
    stderr_redirect = redirect_stderr(StringIO()) if stderr else redirect_stderr(sys.stderr)

    with stdout_redirect, stderr_redirect:
        yield
