"""
conftest.py

This file is used to configure pytest for the project.
"""

import warnings
import os
import pytest
from provada.utils.log import get_logger, setup_logger, logging


# Collect the logger
setup_logger(
    log_filename="pytest",
    verbose=True,
)
logger = get_logger(__name__)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for tests (e.g., '0', '1', '0,1')",
    )
    parser.addoption(
        "--skip-gpu",
        action="store_true",
        default=False,
        help="Skip tests that require GPU",
    )
    parser.addoption(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Run only tests that require GPU",
    )
    parser.addoption(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging during tests",
    )


def pytest_configure(config):
    """Configure pytest with custom settings and register markers."""

    # Register the GPU marker
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring GPU to run"
    )

    # Configure CUDA device if specified
    cuda_device = config.getoption("--device")
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        print(f"\nSetting CUDA_VISIBLE_DEVICES={cuda_device}")

    # Disable Weights & Biases if requested
    no_wandb = config.getoption("--no-wandb")
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        print("\nDisabling Weights & Biases (--no-wandb flag used)")

    # Filter out third-party library warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="esm.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb.*")

    # Filter out third party logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on GPU-related command-line options."""
    skip_gpu = config.getoption("--skip-gpu")
    gpu_only = config.getoption("--gpu-only")

    if skip_gpu and gpu_only:
        raise pytest.UsageError("Cannot use --skip-gpu and --gpu-only together")

    skip_gpu_marker = pytest.mark.skip(reason="Skipping GPU tests (--skip-gpu flag used)")
    skip_non_gpu_marker = pytest.mark.skip(reason="Skipping non-GPU tests (--gpu-only flag used)")

    for item in items:
        has_gpu_marker = "requires_gpu" in item.keywords

        if skip_gpu and has_gpu_marker:
            item.add_marker(skip_gpu_marker)
        elif gpu_only and not has_gpu_marker:
            item.add_marker(skip_non_gpu_marker)


# Optionally, log test failures
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Creates a report for each failed test containing the traceback details.
    Adds traceback details to the log.
    """
    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

    # Log the failure with traceback if the test fails
    if report.when == "call" and report.failed:
        logger.error("Test failed: %s\n%s", item.name, report.longreprtext)


def pytest_runtest_call(item):
    print("-----------")
