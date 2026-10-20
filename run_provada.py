"""
run_provada.py

Main executable for ProVADA
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from onefig import ConfigError
from provada.config import load_config_from_args
from provada.utils.setup import init_wandb
from provada.base_variant import BaseVariant
from provada.utils.log import setup_logger
from provada.sampler.sampler import get_sampler_class


def main():
    """
    Main script for ProVADA

    ``python run_provada.py --config <name_or_path> [field=value ...] [--show] [--help]``

    Overrides are applied after the config file is read. ``--show`` prints the
    resolved config and exits; ``--help`` prints the schema. A configuration
    error is reported as onefig's field-tree summary, without a traceback.
    """

    # ===== ProVADA run setup =====
    try:
        config = load_config_from_args(get_parser())
    except ConfigError as error:
        raise SystemExit(str(error)) from None  # onefig renders its own report
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from None

    # Nothing should mutate the config once the run name is derived from it.
    run_name = config.resolve_run_name()
    config.freeze()

    setup_logger(
        verbose=config.verbose,
        log_filename=run_name,
        logging_subdir=config.logging_subdir,
    )
    config.display(run_name)

    wandb_run = init_wandb(
        no_wandb=config.no_wandb,
        run_name=run_name,
        config=config,
        wandb_project=config.wandb_project,
    )

    # ===== Base Variant Setup =====
    base_variant = BaseVariant()
    base_variant.load_base_variant_from_config(config.base_variant)

    # ===== Sampler Setup =====
    SamplerClass = get_sampler_class(name=config.sampler.sampler_type)
    sampler = SamplerClass(
        base_variant=base_variant,
        num_iters=config.sampler.num_iters,
        population_size=config.sampler.population_size,
        top_k_fraction=config.sampler.top_k_fraction,
        score_weights=config.sampler.score_weights,
        generator_type=config.generator.generator_type,
        generation_kwargs=config.generator.kwargs(),
        masking_schedule_config=config.masking_schedule,
        temperature_schedule_config=config.temperature_schedule,
        seed=config.seed,
        run_name=run_name,
        output_dir=config.output_dir,
        reference_distribution_file=config.reference_distribution_file,
        wandb_run=wandb_run,
        config=config,
        greedy=config.sampler.greedy,
        normalize=config.sampler.normalize,
        resampling_method=config.sampler.resampling_method,
    )

    sampler.run()


def get_parser():
    """
    Collects arguments for run_provada.py

    Only ``--config`` is declared. Every other setting is a field on
    ``ProvadaConfig``, so it is overridden on the command line rather than
    duplicated here as a flag — which keeps the CLI in step with the schema
    instead of drifting from it.

    ``add_help=False`` hands ``--help`` to onefig, which lists every overridable
    field with its type, default and description. argparse would otherwise
    intercept it and print a listing containing only ``--config``.
    """
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Run ProVADA from a config file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config file name or path",
        required=True,
    )
    return parser


if __name__ == "__main__":
    main()
