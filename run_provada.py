"""
run_provada.py

Main executable for ProVADA
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from provada.utils.setup import (
    get_config_from_args,
    get_run_name_from_config,
    init_wandb,
)
from provada.base_variant import BaseVariant
from provada.utils.log import setup_logger
from provada.sampler.sampler import get_sampler_class
from provada.utils.env import device_manager


def main():
    """
    Main script for ProVADA
    """

    # ===== ProVADA run setup =====
    # Create the config
    parser = get_parser()
    # Get config from args
    config = get_config_from_args(parser)
    if not hasattr(config, "device") or config.device is None:
        config.device = device_manager.get_device()
    if not hasattr(config, "reference_distribution_file"):
        config.reference_distribution_file = None

    # Set the run name
    run_name = get_run_name_from_config(config, script_name="run_provada")
    # Set the logger
    setup_logger(
        verbose=config.verbose,
        log_filename=run_name,
        logging_subdir=getattr(config, "logging_subdir", None),
    )

    # Create wandb run
    wandb_run = init_wandb(
        no_wandb=config.no_wandb,
        run_name=run_name,
        config=config,
        wandb_project=config.wandb_project,
    )

    # ===== Base Variant Setup =====
    # Load the base variant
    base_variant = BaseVariant()
    base_variant.load_base_variant_from_config(config.base_variant)

    # ===== Sampler Setup =====

    # Load the sampler
    SamplerClass = get_sampler_class(name=config.sampler.sampler_type)
    sampler = SamplerClass(
        base_variant=base_variant,
        num_iters=config.sampler.num_iters,
        population_size=config.sampler.population_size,
        top_k_fraction=config.sampler.top_k_fraction,
        score_weights=config.sampler.score_weights,
        generator_type=config.generator.generator_type,
        generation_kwargs=config.generator.generation_kwargs,
        masking_strategy_type=config.masking_strategy.masking_strategy_type,
        masking_strategy_kwargs=config.masking_strategy.masking_strategy_kwargs,
        masking_schedule_config=config.masking_schedule_config,
        temperature_schedule_config=config.temperature_schedule_config,
        seed=config.seed,
        run_name=run_name,
        device=config.device,
        reference_distribution_file=config.reference_distribution_file,
        wandb_run=wandb_run,
        config=config,
    )

    sampler.run()


def get_parser():
    """
    Collects arguments for run_provada.py

    NOTE: Default values should be specified as None. These will be overridden by
    the specified config file.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config file name",
        required=True,
    )

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--reference_distribution_file", type=str, default=None)
    parser.add_argument("--logging_subdir", type=str, default=None)

    return parser


if __name__ == "__main__":
    main()
