"""
env.py

Contains utility functions related to the environment
"""

import time
import torch
import sys
import os
import fcntl
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from typing import Union
from provada.utils.log import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """
    Manages the global device for the entire codebase.
    """

    _instance = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_device(self, device: str):
        """Set the global device for the entire codebase"""
        self._device = get_device_string(device)

    def get_device(self) -> str:
        """Get the current global device"""
        if self._device is not None:
            return self._device
        # Fallback to auto-detection
        return "cuda:0" if torch.cuda.is_available() else "cpu"


device_manager = DeviceManager()


def number_of_available_gpus():
    """
    Returns the number of available GPUs.
    """
    return torch.cuda.device_count()


def number_of_available_cpus():
    """
    Returns the number of available CPUs.
    """
    return os.cpu_count()


def parse_cuda_device_index(device_string: str):
    """
    Returns the integer index of the GPU specified by a cuda device string.
    """

    # If the device is not a cuda device string, raise an error
    if not device_string.startswith("cuda"):
        raise ValueError("Device string must start with 'cuda'")

    # If the device is "cuda", return 0
    if device_string == "cuda":
        return 0

    # Otherwise, return the integer index of the GPU
    device_string = device_string.replace("cuda:", "")
    device_int = int(device_string)

    if device_int >= number_of_available_gpus():
        raise ValueError(
            f"Device index {device_int} is greater than the number of available GPUs ({number_of_available_gpus()})"
        )

    return device_int


def get_device_string(device_str_or_int: Union[int, str, torch.device]):
    """
    Returns the string representation of the GPU specified by an integer index.
    """
    # If the device is a torch.device, get the string representation
    if isinstance(device_str_or_int, torch.device):
        device_str_or_int = str(device_str_or_int)

    # If we have a string
    if isinstance(device_str_or_int, str):
        # If it's just "cuda", return "cuda:0"
        if device_str_or_int == "cuda":
            return "cuda:0"

        if device_str_or_int == "cpu":
            return "cpu"

        # Otherwise, ensure it parses correctly to a single integer
        try:
            device_int = parse_cuda_device_index(device_str_or_int)
            return f"cuda:{device_int}"
        except ValueError:
            raise ValueError(f"Invalid device string: {device_str_or_int}")

    # If it's an integer, return the string representation
    elif isinstance(device_str_or_int, int):
        return f"cuda:{device_str_or_int}"

    else:
        raise ValueError(f"Invalid device: {device_str_or_int}")


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


class LockTimeout(Exception):
    """Raised when lock acquisition times out."""

    pass


@contextmanager
def lock_via_lockfile(lock_file_path: Path, timeout: int = 30):
    """
    Acquire an exclusive lock via a lock file with timeout.

    Args:
        lock_file_path: Path to the lock file
        timeout: Maximum time to wait for lock in seconds

    Raises:
        LockTimeout: If lock cannot be acquired within timeout period

    Example:
        try:
            with lock_via_lockfile(Path("my.lock"), timeout=30):
                # Critical section - lock is held here
                initialize_resource()
        except LockTimeout:
            print("Could not acquire lock in time")
    """
    # Ensure parent directory exists
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)

    f = None
    try:
        f = open(lock_file_path, "w")
        start_time = time.time()

        # Try to acquire lock with timeout
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Lock acquired successfully
                yield
                return
            except BlockingIOError:
                # Lock is held by another process, wait a bit
                time.sleep(1)

        # Timeout occurred
        raise LockTimeout(
            f"Could not acquire lock on {lock_file_path} within {timeout} seconds"
        )

    finally:
        # Always release lock and close file
        if f is not None:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except:
                pass
            f.close()
