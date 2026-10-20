"""
cache.py

Helper functions for caching data
"""

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore
from typing import Dict
from provada.utils.log import get_logger

logger = get_logger(__name__)


class CSVCache:
    """
    Class to cache data to a csv file.
    """

    def __init__(self, csv_path: str, max_pending_writes: int = 5):
        """
        Initialize the CSVCache

        Args:
            csv_path: The path to the csv file to cache data to
            max_pending_writes: The maximum number of pending writes to allow
                before blocking.
        """
        self.csv_path = csv_path
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.semaphore = Semaphore(max_pending_writes)
        self._error_lock = Lock()
        self._write_error = None

        logger.info(f"Initialized CSVCache at {csv_path}")

    def cache_df(self, df: pd.DataFrame):
        """
        Caches a dataframe to a csv file. If the csv file already exists, the
        entries are appended to the file.

        NOTE: This function assumes that the dataframe passed in has the same
        columns as the csv file in the same order. Only include new rows that
        are not already in the csv file each call.
        """

        self._raise_write_error()

        # Block if too many writes are pending for this specific path.
        self.semaphore.acquire()

        try:
            # A preceding write may have failed while this producer was waiting.
            self._raise_write_error()
            self.executor.submit(self._write_csv_impl, df.copy())
        except BaseException:
            # No worker owns this slot when copying or submission fails.
            self.semaphore.release()
            raise

    def _raise_write_error(self):
        with self._error_lock:
            error = self._write_error
        if error is not None:
            raise error

    def _write_csv_impl(self, df: pd.DataFrame):
        try:
            # Ensure directory exists
            if os.path.dirname(self.csv_path) != "":
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

            # If the file does not exist, create it
            if not os.path.exists(self.csv_path):
                df.to_csv(self.csv_path, index=False)
            else:
                # Otherwise, append the entries to the file
                df.to_csv(self.csv_path, index=False, header=False, mode="a")
        except BaseException as error:
            # Retain only the first failure, rather than every completed future.
            with self._error_lock:
                if self._write_error is None:
                    error.add_note(f"CSV write failed for {self.csv_path}")
                    self._write_error = error
            raise
        finally:
            self.semaphore.release()

    def shutdown(self):
        self.executor.shutdown(wait=True)
        self._raise_write_error()


# Global cache manager
_path_caches: Dict[str, CSVCache] = {}


def cache_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Creates a separate cache/queue for each unique path.
    Each path has its own background thread and semaphore.
    """
    # Cast any boolean columns to int to save space
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Create cache for this path if it doesn't exist
    if path not in _path_caches:
        _path_caches[path] = CSVCache(path)

        if os.path.exists(path):
            # Remove the file if it exists
            os.remove(path)

    # Use the cache specific to this path
    _path_caches[path].cache_df(df)


def shutdown_cache():
    """Shutdown all path-specific caches"""
    errors = []
    try:
        for path, cache in list(_path_caches.items()):
            try:
                cache.shutdown()
            except BaseException as error:
                errors.append((path, error))
    finally:
        _path_caches.clear()
    if errors:
        path, error = errors[0]
        error.add_note(f"CSV cache shutdown failed for {path}")
        for other_path, other_error in errors[1:]:
            error.add_note(f"Additional cache shutdown failure for {other_path}: {other_error}")
        raise error
