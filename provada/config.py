"""
provada/config.py

Typed configuration schema for ProVADA, built on onefig (OmegaConf-style YAML
loading plus Pydantic v2 validation).

Two things worth knowing when writing a config:

- Every component parameter is a declared field on a discriminated-union
  variant, so a typo or an out-of-range value fails at config load with a
  field-tree error rather than surfacing deep inside a run.
- Each union's discriminator (``generator_type``, ``sampler_type``, ...) selects
  exactly one variant, so only that variant's fields are accepted.

Runtime values that components need but configs must not supply (sequence length,
fixed positions, total iteration count) are injected by the sampler, not declared
here.
"""

from argparse import ArgumentParser
from math import isfinite
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional, Union

from onefig import ConfigModel, tagged_union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============================================================================
# Base variant
# ============================================================================


class BaseVariantConfig(BaseModel):
    """Inputs describing the starting protein."""

    model_config = ConfigDict(extra="forbid")

    sequence_file_path: str = Field(
        description="Path to a text file holding the wild-type sequence.",
    )
    structure_file_path: Optional[str] = Field(
        default=None,
        description="Path to a PDB/CIF structure. Required by structure-conditioned generators.",
    )
    fixed_position_files: List[str] = Field(
        default_factory=list,
        description="Files listing 1-indexed positions to hold fixed during design.",
    )
    ensemble_structure_dir: Optional[str] = Field(
        default=None,
        description="Directory of conformers to score against, instead of one structure.",
    )
    main_conformer: Optional[str] = Field(
        default=None,
        description="Conformer in ensemble_structure_dir to treat as the reference.",
    )
    mhc_alleles: Optional[str] = Field(
        default=None,
        description="Comma-separated MHC alleles for immunogenicity scoring.",
    )
    exclude_self_peptides: bool = Field(
        default=False,
        description="Skip peptides the human proteome already carries when scoring immunogenicity.",
    )
    self_peptide_db_path: Optional[str] = Field(
        default=None,
        description=(
            "File of peptides to treat as self, one per line, optionally gzipped. "
            "Defaults to the managed human 9mer peptidome."
        ),
    )

    @model_validator(mode="after")
    def _self_peptide_db_needs_exclusion(self) -> "BaseVariantConfig":
        """A database is only read when exclusion is asked for, so say so."""
        if self.self_peptide_db_path is not None and not self.exclude_self_peptides:
            raise ValueError(
                "self_peptide_db_path is set but exclude_self_peptides is false, so the "
                "database would never be read; set exclude_self_peptides: true"
            )
        return self

    @model_validator(mode="after")
    def _ensemble_needs_both_fields(self) -> "BaseVariantConfig":
        """An ensemble is only usable when its reference conformer is named."""
        has_dir = self.ensemble_structure_dir is not None
        has_main = self.main_conformer is not None
        if has_dir != has_main:
            raise ValueError(
                "ensemble_structure_dir and main_conformer must be set together; "
                f"got ensemble_structure_dir={self.ensemble_structure_dir!r}, "
                f"main_conformer={self.main_conformer!r}"
            )
        return self


# ============================================================================
# Generators
# ============================================================================


class GeneratorVariant(BaseModel):
    """Base for generator config variants.

    Fields on a variant are forwarded to the generator's ``generate()`` call;
    the discriminator itself selects the generator from the registry.
    """

    model_config = ConfigDict(extra="forbid")

    def kwargs(self) -> Dict[str, Any]:
        """Return generation keyword arguments, excluding the discriminator."""
        return self.model_dump(exclude={"generator_type"})


class _MaskedModelGeneratorConfig(GeneratorVariant):
    """Shared parameters for the masked-model (ESM) generators.

    proto-tools gives ESM2 and ESM3 one sampling surface, so these are the same
    for both. ``schedule``, ``strategy`` and ``num_steps`` apply only to
    ``sampling_method="iterative_refinement"``.
    """

    max_batch_size: int = Field(default=20, gt=0)
    sampling_method: Literal["iterative_refinement", "single_pass"] = Field(
        default="iterative_refinement",
        description="Multi-step decoding, or fill every position in one pass.",
    )
    schedule: Literal["cosine", "linear"] = "cosine"
    strategy: Literal["random", "entropy"] = "entropy"
    num_steps: int = Field(default=16, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    temperature_annealing: bool = True
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    excluded_amino_acids: Optional[List[str]] = Field(
        default=None,
        description="Single-letter codes never sampled at masked positions.",
        examples=[["C"]],
    )


class ESM3GeneratorConfig(_MaskedModelGeneratorConfig):
    """ESM3 masked-infilling generator."""

    generator_type: Literal["esm3"] = "esm3"
    model_checkpoint: str = Field(
        default="esm3_sm_open_v1", description="ESM3 weights to load."
    )


class ESM2GeneratorConfig(_MaskedModelGeneratorConfig):
    """ESM2 masked-infilling generator."""

    generator_type: Literal["esm2"] = "esm2"
    model_checkpoint: str = Field(
        default="esm2_t33_650M_UR50D", description="ESM2 weights to load."
    )


class _MPNNFamilyConfig(GeneratorVariant):
    """Shared parameters for the ProteinMPNN generators.

    Neither the weight set nor the implementation is a parameter: both are
    chosen by the discriminator, since each registered generator sets its own
    ``_WEIGHTS`` and ``_TOOLKIT``. The structure is supplied by the base variant
    through ``get_extra_kwargs``.
    """

    generations_per_sequence: int = Field(default=10, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    excluded_amino_acids: Optional[List[str]] = Field(
        default=None,
        description="Amino acids the generator may not place, e.g. ['C'].",
    )


class MPNNGeneratorConfig(_MPNNFamilyConfig):
    """ProteinMPNN with the original weight set, through the JAX toolkit."""

    generator_type: Literal["mpnn"] = "mpnn"


class SolubleMPNNGeneratorConfig(_MPNNFamilyConfig):
    """ProteinMPNN with the soluble-protein weight set, through the JAX toolkit."""

    generator_type: Literal["soluble_mpnn"] = "soluble_mpnn"


class AbMPNNGeneratorConfig(_MPNNFamilyConfig):
    """ProteinMPNN with the antibody-optimized weight set.

    AbMPNN loads through ColabDesign, so it has no PyTorch counterpart.
    """

    generator_type: Literal["abmpnn"] = "abmpnn"


class MPNNPytorchGeneratorConfig(_MPNNFamilyConfig):
    """ProteinMPNN with the original weight set, through the PyTorch toolkit.

    Same weights as ``mpnn``, different implementation: this one conditions on
    every non-masked residue of the prompt rather than only the fixed
    positions, so it does not produce the same designs.
    """

    generator_type: Literal["mpnn_pytorch"] = "mpnn_pytorch"


class SolubleMPNNPytorchGeneratorConfig(_MPNNFamilyConfig):
    """solubleMPNN through the PyTorch toolkit."""

    generator_type: Literal["soluble_mpnn_pytorch"] = "soluble_mpnn_pytorch"


class _CalibyFamilyConfig(GeneratorVariant):
    """Shared parameters for the Caliby generators.

    The weight set is not a parameter: it is chosen by the discriminator, since
    each registered generator sets its own checkpoint. The structure, and any
    conformer ensemble, come from the base variant through ``get_extra_kwargs``.
    """

    generations_per_sequence: int = Field(default=10, gt=0)
    temperature: float = Field(default=0.01, gt=0.0)
    excluded_amino_acids: Optional[List[str]] = Field(
        default=None,
        description="Amino acids the generator may not place, e.g. ['C'].",
    )


class CalibyGeneratorConfig(_CalibyFamilyConfig):
    """Caliby inverse-folding generator."""

    generator_type: Literal["caliby"] = "caliby"


class SolubleCalibyGeneratorConfig(_CalibyFamilyConfig):
    """Caliby with the soluble-protein weight set."""

    generator_type: Literal["soluble_caliby"] = "soluble_caliby"


class RandomGeneratorConfig(GeneratorVariant):
    """Codon-scheme random infilling, for baselines and tests."""

    generator_type: Literal["random"] = "random"
    codon_scheme: str = "NNN"


if TYPE_CHECKING:
    GeneratorConfig = Union[
        ESM3GeneratorConfig,
        ESM2GeneratorConfig,
        MPNNGeneratorConfig,
        SolubleMPNNGeneratorConfig,
        AbMPNNGeneratorConfig,
        MPNNPytorchGeneratorConfig,
        SolubleMPNNPytorchGeneratorConfig,
        CalibyGeneratorConfig,
        SolubleCalibyGeneratorConfig,
        RandomGeneratorConfig,
    ]
else:
    GeneratorConfig = tagged_union(
        ESM3GeneratorConfig,
        ESM2GeneratorConfig,
        MPNNGeneratorConfig,
        SolubleMPNNGeneratorConfig,
        AbMPNNGeneratorConfig,
        MPNNPytorchGeneratorConfig,
        SolubleMPNNPytorchGeneratorConfig,
        CalibyGeneratorConfig,
        SolubleCalibyGeneratorConfig,
        RandomGeneratorConfig,
        tag="generator_type",
    )


# ============================================================================
# Masking strategies
# ============================================================================


# ============================================================================
# Schedules
# ============================================================================


class ScheduleVariant(BaseModel):
    """Base for schedule config variants.

    ``total_iters`` is supplied by the sampler from ``sampler.num_iters``.
    """

    model_config = ConfigDict(extra="forbid")

    def kwargs(self) -> Dict[str, Any]:
        """Return constructor keyword arguments, excluding the discriminator."""
        return self.model_dump(exclude={"schedule_type"})


class ConstantScheduleConfig(ScheduleVariant):
    """Hold a single value for every iteration."""

    schedule_type: Literal["constant"] = "constant"
    value: float


class LinearScheduleConfig(ScheduleVariant):
    """Interpolate linearly from ``start_value`` to ``stop_value``."""

    schedule_type: Literal["linear"] = "linear"
    start_value: float
    stop_value: float


class GeometricScheduleConfig(ScheduleVariant):
    """Interpolate geometrically from ``start_value`` to ``stop_value``."""

    schedule_type: Literal["geometric"] = "geometric"
    start_value: float = Field(gt=0.0)
    stop_value: float = Field(gt=0.0)


class PowerScheduleConfig(ScheduleVariant):
    """Interpolate as ``start + (stop - start) * frac**alpha``."""

    schedule_type: Literal["power"] = "power"
    start_value: float
    stop_value: float
    alpha: float = Field(gt=0.0)


if TYPE_CHECKING:
    MaskingScheduleConfig = Union[
        ConstantScheduleConfig,
        LinearScheduleConfig,
        GeometricScheduleConfig,
        PowerScheduleConfig,
    ]
    # A constant Metropolis-Hastings temperature defeats annealing, but it is
    # how baseline runs are done, so it is permitted and warned about at
    # startup rather than rejected here.
    TemperatureScheduleConfig = Union[
        ConstantScheduleConfig,
        LinearScheduleConfig,
        GeometricScheduleConfig,
        PowerScheduleConfig,
    ]
else:
    MaskingScheduleConfig = tagged_union(
        ConstantScheduleConfig,
        LinearScheduleConfig,
        GeometricScheduleConfig,
        PowerScheduleConfig,
        tag="schedule_type",
    )
    TemperatureScheduleConfig = tagged_union(
        ConstantScheduleConfig,
        LinearScheduleConfig,
        GeometricScheduleConfig,
        PowerScheduleConfig,
        tag="schedule_type",
    )


# ============================================================================
# Sampler
# ============================================================================

SamplerType = Literal["rejection", "mada"]


class SamplerConfig(BaseModel):
    """Search-loop parameters shared by every sampler."""

    model_config = ConfigDict(extra="forbid")

    sampler_type: SamplerType = Field(
        default="mada",
        description="Sampler implementation; mada uses greedy to select its policy.",
    )
    greedy: bool = Field(
        default=True,
        description="Enable top-k selection and weighted population reconstruction in mada.",
    )
    normalize: bool = Field(
        default=True,
        description="Use fixed bounds to scale metrics and composite fitness to [0,1]. "
        "False combines oriented raw metrics. Reference improvement is saved separately.",
    )
    resampling_method: Literal["systematic", "multinomial"] = Field(
        default="systematic",
        description="Resampling method for non-greedy SMC.",
    )
    num_iters: int = Field(default=100, gt=0)
    population_size: int = Field(default=5, gt=0)
    top_k_fraction: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Elite fraction for greedy selection; stochastic sampling ignores "
        "this setting and uses K=population_size.",
    )
    score_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Score key to weight. Keys select which evaluators run.",
    )

    @field_validator("score_weights")
    @classmethod
    def _require_a_nonzero_weight(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Reject weightings that cannot produce a fitness signal.

        Score keys themselves are validated by the evaluator registry at sampler
        startup; resolving them here would pull the model stack into config import.
        """
        if not value:
            raise ValueError("score_weights must name at least one score")
        if any(not isfinite(weight) or weight < 0 for weight in value.values()):
            raise ValueError("score_weights must be finite and nonnegative; score metadata sets direction")
        if all(weight == 0 for weight in value.values()):
            raise ValueError(
                "score_weights are all zero, so every variant would score identically"
            )
        return value

    @model_validator(mode="after")
    def _validate_greedy_option(self) -> "SamplerConfig":
        if self.sampler_type != "mada" and "greedy" not in self.model_fields_set:
            self.greedy = False
        if self.greedy and self.sampler_type != "mada":
            raise ValueError(
                "greedy is an option on mada; use sampler_type: "
                "mada with greedy: true, or select another sampler explicitly"
            )
        return self


# ============================================================================
# Top level
# ============================================================================


class ProvadaConfig(ConfigModel):
    """Top-level ProVADA run configuration."""

    seed: int = 42
    verbose: bool = False

    no_wandb: bool = Field(default=False, description="Disable Weights & Biases logging.")
    wandb_project: str = Field(
        default="provada", description="W&B project to log this run under."
    )

    reference_distribution_file: Optional[str] = Field(
        default=None,
        description="CSV of reference scores used to set unbounded score ranges. "
        "Generated and cached in the run's output directory when absent.",
    )
    logging_subdir: Optional[str] = Field(
        default=None, description="Subdirectory of logs/ to write this run's log into."
    )
    output_dir: Optional[str] = Field(
        default=None, description="Root for run outputs. Defaults to results/."
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Name for this run. Unset derives a stable name from the config.",
    )

    base_variant: BaseVariantConfig
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    generator: GeneratorConfig = Field(default_factory=RandomGeneratorConfig)
    masking_schedule: Optional[MaskingScheduleConfig] = Field(
        default=None,
        description="Fraction of designable positions to mask per iteration. "
        "Defaults to a constant 0.3 schedule.",
    )
    temperature_schedule: Optional[TemperatureScheduleConfig] = Field(
        default=None,
        description="Metropolis-Hastings temperature per iteration. "
        "Defaults to a power schedule from 5.0 to 0.1.",
    )

    #: Top-level fields that describe how a run executes rather than what it
    #: computes. Excluded from both the run name and the logged hyperparameters,
    #: so the same experiment keeps one identity across devices, verbosity and
    #: logging choices. ``run_name`` is itself excluded: naming a run must not
    #: change what the name is derived from.
    NON_IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "verbose",
        "no_wandb",
        "logging_subdir",
        "output_dir",
        "run_name",
    )

    def resolve_run_name(self) -> str:
        """Return this run's name: ``run_name`` if set, else derived from the config.

        When derived it is content-addressable — the same experiment always
        produces the same name, regardless of the execution-only fields in
        ``NON_IDENTITY_FIELDS``. The trailing digest keeps two configs that
        happen to draw the same codename from colliding on an output directory.
        """
        if self.run_name:
            return self.run_name
        codename = self.codename(
            exclude=self.NON_IDENTITY_FIELDS, suffix=True, suffix_length=5
        )
        return f"{self.config_name or 'run'}_{codename}"

    def identity_config(self) -> Dict[str, Any]:
        """Return the flattened config minus execution-only fields.

        The same subset the run name is derived from, so logged hyperparameters
        describe the experiment's identity. Without this, a sweep's parameter
        space and parallel-coordinates plots are cluttered with the device, the
        output path and the logging toggles.
        """
        return {
            key: value
            for key, value in self.to_flat_dict().items()
            if key.split(".", 1)[0] not in self.NON_IDENTITY_FIELDS
        }


# ============================================================================
# CLI
# ============================================================================


#: Tokens onefig's override parser handles itself. ``--help`` reaches it only
#: because the parser is built with ``add_help=False``; argparse would otherwise
#: consume it and print its own much thinner flag listing.
#: The completion-install tokens are deliberately absent: they take a following
#: value, and the ``onefig`` console command is the place to run them.
ONEFIG_PASSTHROUGH_TOKENS = frozenset({"--show", "--help", "-h"})


def normalize_cli_tokens(tokens: List[str]) -> List[str]:
    """Rewrite argparse-style tokens into the ``key=value`` form onefig expects.

    onefig's CLI override parser accepts only bare ``key=value`` tokens, but
    ProVADA is driven by tools that emit argparse style: ``run_multiple.py``
    forwards ``--sampler_type mada``, and a W&B sweep agent passes
    ``--key=value``. This normalizes all three accepted forms:

    ``--key value`` and ``--key=value`` become ``key=value``; a bare ``--flag``
    with no following value becomes ``flag=true``. Hyphens inside a key become
    underscores so ``--top-k-fraction`` reaches ``top_k_fraction``.

    Values are left as strings; onefig JSON-coerces them against the target
    field's type.

    Args:
        tokens (List[str]): Raw CLI tokens, e.g. from ``parse_known_args()``.

    Returns:
        List[str]: Tokens in ``key=value`` form.

    Raises:
        ValueError: If a token is neither ``--``-prefixed nor already ``key=value``.
    """
    normalized: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]

        # onefig consumes these itself; rewriting them into key=value would
        # turn them into unknown config keys.
        if token in ONEFIG_PASSTHROUGH_TOKENS:
            normalized.append(token)
            index += 1
            continue

        if not token.startswith("--"):
            if "=" in token:
                normalized.append(token)
                index += 1
                continue
            raise ValueError(
                f"Unrecognized argument {token!r}: expected '--key value', "
                "'--key=value', or 'key=value'."
            )

        body = token[2:]
        if "=" in body:
            key, value = body.split("=", 1)
            normalized.append(f"{key.replace('-', '_')}={value}")
            index += 1
            continue

        key = body.replace("-", "_")
        # The next token is this key's value unless it is itself a flag. Testing
        # for "--" rather than "-" keeps negative numbers usable as values, so
        # `--start_value -0.5` is a value and not a flag followed by garbage.
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            normalized.append(f"{key}={tokens[index + 1]}")
            index += 2
        else:
            normalized.append(f"{key}=true")
            index += 1

    return normalized


#: Discriminator field of each tagged union, mapped to the config field it
#: selects a variant for. ``schedule_type`` is shared by the masking and
#: temperature schedules, so overriding it needs a dotted path
#: (``masking_schedule.schedule_type=power``); onefig rejects the bare leaf
#: as ambiguous.
UNION_FIELD_BY_DISCRIMINATOR = {
    "generator_type": "generator",
}


def _reject_variant_switches(config: ProvadaConfig, tokens: List[str]) -> None:
    """Raise if a token would switch a tagged union to a different variant.

    Overrides are merged field-by-field into the variant already installed on
    the config, so switching one leaves the outgoing variant's fields behind and
    the incoming variant rejects them as unknown. Rather than surface that as a
    confusing error about an unrelated field, refuse up front and point at the
    supported route.

    Args:
        config (ProvadaConfig): The config overrides will be applied to.
        tokens (List[str]): Tokens in ``key=value`` form.

    Raises:
        ValueError: If a discriminator override names a different variant.
    """
    for token in tokens:
        key, _, value = token.partition("=")
        leaf = key.rsplit(".", 1)[-1]
        field_name = UNION_FIELD_BY_DISCRIMINATOR.get(leaf)
        if field_name is None:
            continue

        current = getattr(getattr(config, field_name, None), leaf, None)
        if current is not None and value != current:
            raise ValueError(
                f"Cannot switch {field_name!r} from {current!r} to {value!r} on the "
                f"command line: each variant has its own fields, and the ones already "
                f"set for {current!r} do not exist on {value!r}. Use a config file "
                f"with the {value!r} settings instead (see inputs/renin/ for one "
                f"config per generator)."
            )


def load_config_from_args(
    parser: ArgumentParser, argv: Optional[List[str]] = None
) -> ProvadaConfig:
    """Build a validated :class:`ProvadaConfig` from declared and ad-hoc CLI args.

    Resolution order, lowest precedence first: the YAML file named by
    ``--config``, then the parser's declared flags, then any remaining
    ``key=value`` overrides. Declared flags left unset are skipped rather than
    applied, so they never clobber a value the YAML set.

    Args:
        parser (ArgumentParser): Parser declaring at least ``--config``. Flags
            should default to ``None`` so that omitting them is distinguishable
            from passing a falsy value.
        argv (Optional[List[str]]): Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        ProvadaConfig: The validated configuration.

    Raises:
        ValueError: If ``--config`` is absent.
        ConfigError: If the YAML or any override fails validation.
    """
    from provada.paths import REPO_ROOT

    known_args, unknown_args = parser.parse_known_args(argv)

    config_ref = getattr(known_args, "config", None)
    if not config_ref:
        raise ValueError("--config is required to locate a ProVADA config file.")

    # Searching from the repo root (not the cwd) keeps bare config names
    # resolvable regardless of where the run is launched from.
    config = ProvadaConfig.load(config_ref, search_root=REPO_ROOT)

    declared = {
        key: value for key, value in vars(known_args).items() if key != "config"
    }
    config.update_from_args(declared, strict=False, skip_none=True)

    if unknown_args:
        tokens = normalize_cli_tokens(unknown_args)
        _reject_variant_switches(config, tokens)
        config.update_from_cli(tokens, strict=True)

    return config
