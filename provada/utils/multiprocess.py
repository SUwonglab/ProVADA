"""
multiprocess.py

Utility functions for multiprocessing.
"""

import multiprocessing as mp
import threading
import time
from typing import List, Callable, Any
from typing import Tuple, Dict
from provada.utils.log import setup_logger
from tqdm import tqdm

_worker_local = threading.local()


_POOL_SINGLETON = None


def get_pool(n_workers: int = 5):
    global _POOL_SINGLETON
    if _POOL_SINGLETON is None:
        _POOL_SINGLETON = PersistentPool(n_workers=n_workers)
    return _POOL_SINGLETON


def _do_init(init_fn: Callable):
    """
    Run an initialization function inside a worker process.

    Args:
        init_fn: A function that returns a dict of resources (e.g. libraries,
                 models, configs) to be stored in the worker's context.

    Returns:
        True (dummy return value to satisfy map contract).
    """

    if not hasattr(_worker_local, "ctx"):
        _worker_local.ctx = {}
        _worker_local.ctx["success_list"] = []
        _worker_local.ctx["retry_count_list"] = []
        _worker_local.ctx["message_list"] = []

    retry_count = 0
    while retry_count < 3:
        # Attempt the initialization
        ctx = init_fn()

        # If the initialization function returns a dict, check if it has a "success" key
        if isinstance(ctx, dict):
            if ctx.get("success", False):
                break
        retry_count += 1
        time.sleep(1)

    # Add the context to the worker local context
    _worker_local.ctx.update(ctx)

    # Add the success to the success list
    _worker_local.ctx["success_list"].append(ctx.get("success", False))
    _worker_local.ctx["retry_count_list"].append(retry_count)
    _worker_local.ctx["message_list"].append(ctx.get("message", ""))

    setup_logger(suppress_console_logging=True)
    return retry_count < 3


def _do_work(func_args: Tuple[int, Callable, Tuple, Dict]):
    """
    Worker-side function to execute a task.

    Args:
        func_args: Tuple of (func, args, kwargs). `func` must accept an extra
                   `_ctx` argument, which provides access to the worker's context.

    Returns:
        The result of func(*args, **kwargs, _ctx=worker_context).
    """
    idx, func, args, kwargs = func_args
    return idx, func(*args, **kwargs)


def _get_ctx_from_keys(keys: List[str]) -> Dict[str, Any]:
    """
    Worker-side function to retrieve specified keys from worker context.

    Args:
        keys: List of keys to retrieve from the worker's context.

    Returns:
        Dict containing the requested keys and their values from worker context.
    """
    if not hasattr(_worker_local, "ctx"):
        return {
            "success_list": [False],
            "retry_count_list": [0],
            "message_list": ["Worker not initialized"],
        }

    result = {}
    for key in keys:
        result[key] = _worker_local.ctx.get(key, None)

    return result


class PersistentPool:
    """Spawn-safe persistent worker pool with optional worker initialization."""

    def __init__(self, n_workers: int = 6):
        # Initialize context with 'spawn' for CUDA safety
        ctx = mp.get_context("spawn")

        # Set number of workers
        self.n_workers = min(mp.cpu_count(), n_workers)

        # Create worker pool
        self.pool = ctx.Pool(processes=self.n_workers)

        # Track initialization history for worker recovery
        self.init_history = []  # List of (evaluator_name, init_fn) pairs

    def run_init(self, init_fn: Callable, evaluator_name: str = None):
        # Check for dead workers first
        self._check_and_recover_workers()

        # Add to initialization history
        self.init_history.append((evaluator_name, init_fn))

        # Run normal initialization
        self.pool.map(_do_init, [init_fn] * self.pool._processes)

    def process(self, func, iterable_of_args, show_progress: bool = True, chunksize: int = 20):
        """
        Worker pool function that behaves like starmap but uses imap_unordered
        and can optionally show a progress bar.

        Args:
            func: Function to run in parallel. Must accept multiple arguments.
            iterable_of_args: Iterable of tuples, each tuple contains arguments for func.
            show_progress: Whether to show a tqdm progress bar.
            chunksize: Number of tasks per batch for imap_unordered.

        Returns:
            List of results in any order.
        """

        # Determine the number of tasks we will be running
        n_tasks = len(iterable_of_args)

        # Normalize iterable of args
        enumerated_args = []
        for i, item in enumerate(iterable_of_args):
            if isinstance(item, tuple):
                # Args only
                enumerated_args.append((i, item, {}))
            elif isinstance(item, dict):
                # Kwargs only
                enumerated_args.append((i, (), item))
            else:
                # Single arg
                enumerated_args.append((i, (item,), {}))

        # Set up imap
        it = self.pool.imap_unordered(
            _do_work,
            [(i, func, args, kwargs) for i, args, kwargs in enumerated_args],
            chunksize=chunksize,
        )

        # Dispatch work to workers
        results = []

        if show_progress:
            for r in tqdm(it, total=n_tasks, desc=f"Running {func.__name__}"):
                results.append(r)
        else:
            results = list(it)

        # Sort results by index
        results.sort(key=lambda x: x[0])

        # Remove index
        results = [r[1] for r in results]

        return results

    def _check_and_recover_workers(self):
        """
        Check worker health and reinitialize any that died.
        """
        from provada.utils.log import get_logger

        logger = get_logger(__name__)

        status_list = self.get_ctx_status()

        dead_workers = []
        for i, status in enumerate(status_list):
            if (
                not status or status.get("success_list", [False])[-1] == False
            ):  # Dead or failed worker
                dead_workers.append(i)

        if dead_workers and self.init_history:
            logger.warning(
                f"Found {len(dead_workers)} dead/failed workers: {dead_workers}, reinitializing with {len(self.init_history)} previous evaluators..."
            )
            # Re-run all previous initializations on all workers
            # Since we can't target specific workers, reinitialize all workers
            for evaluator_name, init_fn in self.init_history:
                logger.info(f"Re-initializing all workers with {evaluator_name}")
                results = self.pool.map(_do_init, [init_fn] * self.pool._processes)
                failed_count = sum(1 for r in results if not r)
                success_count = sum(1 for r in results if r)
                logger.info(
                    f"Recovery for {evaluator_name}: {success_count} succeeded, {failed_count} failed"
                )
                if failed_count > 0:
                    logger.error(
                        f"Failed to re-initialize {failed_count} workers with {evaluator_name}"
                    )

            # Check if recovery was successful
            recovery_status = self.get_ctx_status()
            still_failed = []
            for i, status in enumerate(recovery_status):
                if not status or status.get("success_list", [False])[-1] == False:
                    still_failed.append(i)

            if still_failed:
                logger.error(f"Worker recovery failed for workers: {still_failed}")
            else:
                logger.info("All workers successfully recovered")

    def get_ctx_status(self):
        """
        Retrieve status keys from the context of each worker.

        Args:
            keys: List of context keys to retrieve from workers.

        Returns:
            List of dicts, one per worker, containing the requested context values.
        """
        # Get the status keys
        status_keys = ["success_list", "retry_count_list", "message_list"]

        return self.pool.map(_get_ctx_from_keys, [status_keys] * self.pool._processes)
