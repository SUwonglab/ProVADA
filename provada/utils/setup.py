"""
setup.py

Contains functions related to setting up ProVADA runs.
"""

import os
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from onefig import flatten

from provada.utils.log import get_logger

if TYPE_CHECKING:
    from provada.config import ProvadaConfig

logger = get_logger(__name__)

DEFAULT_WANDB_PROJECT = "provada"


def init_wandb(
    no_wandb: bool = False,
    run_name: Optional[str] = None,
    config: Optional[Union["ProvadaConfig", Mapping[str, Any]]] = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
):
    """Initialize a Weights & Biases run.

    Args:
        no_wandb (bool): Skip W&B entirely and return ``None``.
        run_name (Optional[str]): Name to log the run under.
        config (Optional[Union[ProvadaConfig, Mapping[str, Any]]]): Config to
            log. A ``ProvadaConfig`` contributes its identity fields to the W&B
            config panel and is attached unflattened as a YAML artifact. A plain
            mapping is flattened as-is, so callers that build a config by hand,
            such as tests, still work.
        wandb_project (str): Project to log under.

    Returns:
        The ``wandb`` module when logging is enabled, otherwise ``None``.

    Raises:
        ValueError: If ``run_name`` is not provided.
    """
    # Check if wandb is disabled via environment variable
    if os.environ.get("WANDB_MODE") == "disabled":
        if not no_wandb:
            logger.warning(
                "WANDB_MODE environment variable is set to 'disabled'. "
                "Overriding no_wandb parameter to True."
            )
        no_wandb = True

    if no_wandb:
        return None

    if run_name is None:
        raise ValueError("run_name is required to initialize a wandb run.")

    import wandb  # pylint: disable=import-outside-toplevel

    # Log the experiment's identity, not its execution settings: device, output
    # paths and logging toggles would otherwise clutter the config panel and a
    # sweep's parameter space.
    if config is None:
        flat_config = None
    elif hasattr(config, "identity_config"):
        flat_config = config.identity_config()
    else:
        flat_config = flatten(dict(config))

    wandb.init(
        config=flat_config,
        project=wandb_project,
        name=run_name,
    )

    # The unflattened config is easier to reuse than the flattened panel. Only a
    # real config can be written back out as loadable YAML.
    if config is not None and hasattr(config, "save_yaml"):
        yaml_file_path = os.path.join(wandb.run.dir, "unflat_config.yaml")
        config.save_yaml(yaml_file_path)
        wandb.save(yaml_file_path)

    return wandb


def seed_everything(seed):
    """
    Seed everything
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
