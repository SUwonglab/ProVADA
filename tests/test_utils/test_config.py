"""
test_config.py

Tests for the typed ProVADA config schema and its CLI override handling.
"""

import argparse

import pytest
from onefig import ConfigError

from provada.config import (
    LinearScheduleConfig,
    ProvadaConfig,
    RandomGeneratorConfig,
    load_config_from_args,
    normalize_cli_tokens,
)
from provada.paths import REPO_ROOT

MOCK_CONFIG = "mock_config"


def make_parser():
    """A parser that still declares flags, exercising the argparse path.

    ``run_provada.py`` itself declares only ``--config`` now, but
    ``load_config_from_args`` still supports declared flags for wrapper scripts,
    so that path stays covered here.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=MOCK_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--no_wandb", action="store_true", default=None)
    return parser


# ============================================================================
# Loading
# ============================================================================


def test_mock_config_loads_and_selects_variants():
    config = ProvadaConfig.load(MOCK_CONFIG, search_root=REPO_ROOT)

    assert config.seed == 7
    assert config.config_name == MOCK_CONFIG
    assert config.sampler.num_iters == 11
    # Each discriminator selects exactly one variant class.
    assert isinstance(config.generator, RandomGeneratorConfig)
    assert isinstance(config.masking_schedule, LinearScheduleConfig)


def test_kwargs_excludes_the_discriminator():
    """Component kwargs must not carry the tag, which is not a constructor arg."""
    config = ProvadaConfig.load(MOCK_CONFIG, search_root=REPO_ROOT)

    assert "generator_type" not in config.generator.kwargs()
    assert "schedule_type" not in config.masking_schedule.kwargs()


def test_sampler_defaults_to_greedy_top_k():
    config = ProvadaConfig.from_dict({
        "base_variant": {"sequence_file_path": "seq.txt"},
        "sampler": {"score_weights": {"dummy_score": 1.0}},
    })
    assert config.sampler.sampler_type == "mada"
    assert config.sampler.greedy is True
    assert config.sampler.resampling_method == "systematic"


@pytest.mark.parametrize("greedy", [True, False])
def test_greedy_policy_can_be_selected_from_cli(greedy):
    config = load_config_from_args(make_parser(), ["--sampler.greedy", str(greedy).lower()])
    assert config.sampler.sampler_type == "mada"
    assert config.sampler.greedy is greedy


def test_fitness_normalization_defaults_on_and_can_be_disabled():
    config = load_config_from_args(make_parser(), [])
    assert config.sampler.normalize is True
    raw = load_config_from_args(make_parser(), ["--sampler.normalize", "false"])
    assert raw.sampler.normalize is False


@pytest.mark.parametrize("method", ["systematic", "multinomial"])
def test_smc_resampling_method_can_be_selected_from_cli(method):
    config = load_config_from_args(make_parser(), ["--sampler.resampling_method", method])
    assert config.sampler.resampling_method == method


def test_unknown_resampling_method_is_rejected_in_config():
    with pytest.raises((ValueError, ConfigError), match="resampling_method"):
        load_config_from_args(make_parser(), ["--sampler.resampling_method", "unknown"])


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -0.1])
def test_nonfinite_or_negative_score_weights_are_rejected(weight):
    with pytest.raises((ValueError, ConfigError), match="finite and nonnegative"):
        ProvadaConfig.from_dict({
            "base_variant": {"sequence_file_path": "seq.txt"},
            "sampler": {"score_weights": {"dummy_score": weight}},
        })


def test_bundled_examples_use_greedy_top_k():
    for path in sorted((REPO_ROOT / "inputs").glob("*/*.yaml")):
        config = ProvadaConfig.load(str(path), search_root=REPO_ROOT)
        assert config.sampler.sampler_type == "mada", str(path)
        assert config.sampler.greedy is True, str(path)


# ============================================================================
# CLI overrides
# ============================================================================


def test_declared_flag_overrides_config():
    config = load_config_from_args(make_parser(), ["--seed", "99"])
    assert config.seed == 99


def test_unset_flags_do_not_clobber_config():
    """An omitted flag must leave the YAML value alone.

    This is why every flag defaults to None, including the store_true ones:
    a False default would silently overwrite ``no_wandb: true`` from the file.
    """
    config = load_config_from_args(make_parser(), [])

    assert config.seed == 7, "omitted --seed overwrote the config value"
    assert config.no_wandb is True, "omitted --no_wandb overwrote the config value"
    assert config.verbose is False


def test_store_true_flag_applies_when_passed():
    config = load_config_from_args(make_parser(), ["--verbose"])
    assert config.verbose is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--num_iters=5"],
        ["--num_iters", "5"],
        ["num_iters=5"],
        ["--sampler.num_iters=5"],
    ],
    ids=["equals", "spaced", "bare", "dotted"],
)
def test_undeclared_overrides_accepted_in_every_cli_form(argv):
    """All three argparse-ish forms must reach the config.

    run_multiple.py forwards ``--sampler_type mada`` and a W&B sweep
    agent passes ``--key=value``, so none of these forms is hypothetical.
    """
    config = load_config_from_args(make_parser(), argv)
    assert config.sampler.num_iters == 5


def test_undeclared_override_reaches_nested_variant_field():
    config = load_config_from_args(make_parser(), ["--codon_scheme=NNK"])
    assert config.generator.codon_scheme == "NNK"


def test_unknown_override_key_is_rejected():
    with pytest.raises(ValueError, match="not_in_config"):
        load_config_from_args(make_parser(), ["--not_in_config=bad_arg"])


def test_switching_generator_variant_from_cli_is_refused_clearly():
    """Switching a union variant on the CLI must fail with a useful message.

    Overrides merge into the installed variant, so the outgoing variant's
    fields would leak into the incoming one. The old untyped config hit the
    same wall later, as an unexpected-keyword TypeError inside generate().
    """
    with pytest.raises(ValueError, match="Cannot switch 'generator'"):
        load_config_from_args(make_parser(), ["--generator_type=esm3"])


def test_redundant_discriminator_override_is_allowed():
    """Naming the variant already in use is a no-op, not an error."""
    config = load_config_from_args(make_parser(), ["--generator_type=random"])
    assert isinstance(config.generator, RandomGeneratorConfig)


def test_missing_config_argument_raises():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    with pytest.raises(ValueError, match="--config is required"):
        load_config_from_args(parser, [])


# ============================================================================
# Validation
# ============================================================================


@pytest.mark.parametrize(
    "overrides",
    [
        {"bogus_top_level": 1},
        {"generator": {"generator_type": "mpnn", "temperatur": 0.3}},
        {"generator": {"generator_type": "not_a_generator"}},
        {"generator": {"generator_type": "esm3", "top_p": 1.5}},
        {"sampler": {"score_weights": {}}},
        {"sampler": {"score_weights": {"dummy_score": 0.0}}},
        {"sampler": {"top_k_fraction": 1.5}},
        {"sampler": {"num_iters": 0}},
    ],
    ids=[
        "unknown-top-level-key",
        "misspelled-generator-field",
        "unknown-generator-type",
        "top_p-above-one",
        "empty-score-weights",
        "all-zero-score-weights",
        "top_k_fraction-above-one",
        "num_iters-zero",
    ],
)
def test_invalid_configs_are_rejected(overrides):
    """Each of these used to fail at run time, or not at all."""
    data = {
        "base_variant": {"sequence_file_path": "seq.txt"},
        "sampler": {"score_weights": {"dummy_score": 1.0}},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value

    with pytest.raises(ConfigError):
        ProvadaConfig.from_dict(data)


def test_constant_temperature_schedule_is_allowed():
    """A constant temperature is permitted, because baselines need it.

    It defeats Metropolis-Hastings annealing, so the sampler warns at startup
    rather than the schema rejecting it. The original code raised here and the
    experimental branch commented that check out to run baselines.
    """
    config = ProvadaConfig.from_dict(
        {
            "base_variant": {"sequence_file_path": "seq.txt"},
            "sampler": {"score_weights": {"dummy_score": 1.0}},
            "temperature_schedule": {"schedule_type": "constant", "value": 1.0},
        }
    )
    assert config.temperature_schedule.schedule_type == "constant"


def test_base_variant_is_required():
    with pytest.raises(ConfigError):
        ProvadaConfig.from_dict({"sampler": {"score_weights": {"dummy_score": 1.0}}})


# ============================================================================
# Run naming
# ============================================================================


def test_run_name_is_stable_and_prefixed_by_config_name():
    first = ProvadaConfig.load(MOCK_CONFIG, search_root=REPO_ROOT).resolve_run_name()
    second = ProvadaConfig.load(MOCK_CONFIG, search_root=REPO_ROOT).resolve_run_name()

    assert first == second, "run name must be content-addressable"
    assert first.startswith(f"{MOCK_CONFIG}_")


def test_run_name_ignores_runtime_only_fields():
    """Changing where a run executes must not rename the experiment."""
    baseline = load_config_from_args(make_parser(), []).resolve_run_name()
    relocated = load_config_from_args(
        make_parser(), ["--device", "cuda:3", "--verbose"]
    ).resolve_run_name()

    assert baseline == relocated


def test_run_name_changes_with_experimental_fields():
    baseline = load_config_from_args(make_parser(), []).resolve_run_name()
    altered = load_config_from_args(make_parser(), ["--num_iters=12"]).resolve_run_name()

    assert baseline != altered


# ============================================================================
# Token normalization
# ============================================================================


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["--key", "value"], ["key=value"]),
        (["--key=value"], ["key=value"]),
        (["key=value"], ["key=value"]),
        (["--flag"], ["flag=true"]),
        (["--top-k-fraction", "0.4"], ["top_k_fraction=0.4"]),
        (["--a=1", "--b", "2", "--c"], ["a=1", "b=2", "c=true"]),
        (["--start_value", "-0.5"], ["start_value=-0.5"]),
    ],
    ids=[
        "spaced",
        "equals",
        "bare",
        "flag",
        "hyphens-to-underscores",
        "mixed",
        "negative-value",
    ],
)
def test_normalize_cli_tokens(tokens, expected):
    assert normalize_cli_tokens(tokens) == expected


def test_normalize_cli_tokens_rejects_bare_word():
    with pytest.raises(ValueError, match="Unrecognized argument"):
        normalize_cli_tokens(["oops"])


def test_show_token_passes_through_unrewritten():
    """onefig consumes --show itself, so it must not become a config key."""
    assert normalize_cli_tokens(["--num_iters=3", "--show"]) == [
        "num_iters=3",
        "--show",
    ]


def test_show_prints_resolved_config_and_exits():
    with pytest.raises(SystemExit) as excinfo:
        load_config_from_args(make_parser(), ["--num_iters=3", "--show"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize("token", ["--help", "-h"])
def test_help_reaches_onefig_and_exits(token):
    """``run_provada.py`` builds its parser with add_help=False for this.

    argparse would otherwise print a listing containing only ``--config``,
    instead of onefig's per-field schema.
    """
    assert normalize_cli_tokens([token]) == [token]
    with pytest.raises(SystemExit) as excinfo:
        load_config_from_args(make_parser(), [token])
    assert excinfo.value.code == 0


# ============================================================================
# Run identity
# ============================================================================


def test_explicit_run_name_short_circuits_derivation():
    config = load_config_from_args(make_parser(), ["run_name=psb_figure_3"])
    assert config.resolve_run_name() == "psb_figure_3"


def test_setting_run_name_does_not_change_derived_identity():
    """``run_name`` is excluded from identity, so naming a run is not a change."""
    baseline = load_config_from_args(make_parser(), []).identity_config()
    named = load_config_from_args(
        make_parser(), ["run_name=psb_figure_3"]
    ).identity_config()

    assert baseline == named


def test_identity_config_drops_execution_only_fields():
    """Hyperparameters logged to W&B must describe the experiment, not the host.

    Otherwise a sweep's parameter space and its parallel-coordinates plots are
    cluttered with the device, output paths and logging toggles.
    """
    config = load_config_from_args(make_parser(), ["--device", "cuda:3", "--verbose"])
    identity = config.identity_config()

    for field in ProvadaConfig.NON_IDENTITY_FIELDS:
        assert not any(
            key == field or key.startswith(f"{field}.") for key in identity
        ), f"{field} should not be logged as a hyperparameter"

    # The things that do define the experiment are still there.
    assert identity["seed"] == 7
    assert identity["sampler.num_iters"] == 11
    assert identity["generator.generator_type"] == "random"


def test_identity_config_tracks_the_run_name_exclusions():
    """The two must stay in step: same exclusions, same notion of identity."""
    baseline = load_config_from_args(make_parser(), [])
    relocated = load_config_from_args(make_parser(), ["--device", "cuda:3"])

    assert baseline.resolve_run_name() == relocated.resolve_run_name()
    assert baseline.identity_config() == relocated.identity_config()
