"""Input, config and output models shared by the two ProteinMPNN toolkits.

Both toolkits take the same inputs and return the same outputs, so a caller can
switch between them by changing which entry point it calls. They apply a masked
prompt differently, and so produce different designs from the same input:

- ``proteinmpnn_jax`` pins the prompt's residue at ``fixed_positions`` only, by
  writing the sequence input and a large bias term. Everything else is free.
- ``proteinmpnn_pytorch`` overrides the whole parsed sequence, so every
  non-masked position becomes conditioning context.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from proto_tools.utils.base_config import BaseConfig, ConfigField
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField
from pydantic import Field

#: Mask character in a prompt, matching proto-tools' own masked-model tools.
MASK_TOKEN = "_"

#: Weight sets both forks can load. ``antibody`` is AbMPNN.
MPNNWeights = ("original", "soluble", "antibody")


class ProteinMPNNSampleInput(BaseToolInput):
    """One structure to design against, with an optional prompt."""

    structure: str = InputField(
        title="Structure",
        description="PDB content, or a path to a PDB/CIF file.",
    )
    prompt: Optional[str] = InputField(
        default=None,
        title="Prompt",
        description="Sequence with '_' at positions to design; chains split by '/'.",
    )
    fixed_positions: Optional[Dict[str, List[int]]] = InputField(
        default=None,
        title="Fixed Positions",
        description="Chain ID to 1-indexed positions held fixed during design.",
    )
    chain_ids: Optional[List[str]] = InputField(
        default=None,
        title="Chain IDs",
        description="Chains to design. None designs every chain in the structure.",
    )


class ProteinMPNNSampleConfigBase(BaseConfig):
    """Sampling parameters shared by both ProteinMPNN toolkits.

    Each tool subclasses this: registration stamps ``tool_key`` onto the
    config class, so two tools sharing one class would leave it carrying only
    the second one's key.
    """

    num_sequences_to_generate: int = ConfigField(
        default=10,
        gt=0,
        title="Sequences To Generate",
        description="How many sequences to sample from the structure.",
    )
    temperature: float = ConfigField(
        default=0.2,
        gt=0.0,
        title="Temperature",
        description="Sampling temperature; lower is more conservative.",
    )
    weights: str = ConfigField(
        default="original",
        title="Weights",
        description="Weight set: original, soluble, or antibody (AbMPNN).",
        examples=["original", "soluble", "antibody"],
    )
    excluded_amino_acids: Optional[List[str]] = ConfigField(
        default=None,
        title="Excluded Amino Acids",
        description="Single-letter codes never placed at designed positions.",
        examples=[["C"]],
    )


class ProteinMPNNScoreInput(BaseToolInput):
    """Sequences to score against one structure."""

    structure: str = InputField(
        title="Structure",
        description="PDB content, or a path to a PDB/CIF file.",
    )
    sequences: Optional[List[str]] = InputField(
        default=None,
        title="Sequences",
        description=(
            "Sequences to score, chains split by '/'. None scores the "
            "structure's own sequence, giving a single score."
        ),
    )
    fixed_positions: Optional[Dict[str, List[int]]] = InputField(
        default=None,
        title="Fixed Positions",
        description="Chain ID to 1-indexed positions excluded from the score.",
    )
    chain_ids: Optional[List[str]] = InputField(
        default=None,
        title="Chain IDs",
        description="Chains to score. None scores every chain in the structure.",
    )


class ProteinMPNNScoreConfigBase(BaseConfig):
    """Scoring parameters shared by both ProteinMPNN toolkits.

    Subclassed per tool for the same reason as
    :class:`ProteinMPNNSampleConfigBase`.
    """

    weights: str = ConfigField(
        default="original",
        title="Weights",
        description="Weight set: original, soluble, or antibody (AbMPNN).",
        examples=["original", "soluble", "antibody"],
    )


class ProteinMPNNSampleOutput(BaseToolOutput):
    """Sampled sequences and their scores, in sampling order."""

    sequences: List[str] = Field(
        default_factory=list,
        title="Sequences",
        description="Sampled sequences, chains joined by '/'.",
    )
    scores: List[float] = Field(
        default_factory=list,
        title="Scores",
        description="Negative average log-likelihood per sequence; lower is better.",
    )

    output_format_options: ClassVar[List[str]] = ["fasta", "json"]
    output_format_default: ClassVar[str] = "fasta"

    def _export_output(self, path: Any, fmt: str) -> None:
        """Write the sampled sequences to ``path``.

        ``path`` arrives without an extension; adding it here is proto-tools'
        convention for an output that exports to a single file.
        """
        path = Path(path).with_suffix(f".{fmt}")
        if fmt == "fasta":
            with open(path, "w") as handle:
                for index, sequence in enumerate(self.sequences):
                    score = self.scores[index] if index < len(self.scores) else float("nan")
                    handle.write(f">sample_{index} score={score:.6f}\n{sequence}\n")
            return
        if fmt == "json":
            import json

            with open(path, "w") as handle:
                json.dump({"sequences": self.sequences, "scores": self.scores}, handle)
            return
        raise ValueError(f"Unsupported output format {fmt!r}")


class ProteinMPNNScoreOutput(BaseToolOutput):
    """One score per input sequence, in input order."""

    scores: List[float] = Field(
        default_factory=list,
        title="Scores",
        description="Negative average log-likelihood per sequence; lower is better.",
    )

    output_format_options: ClassVar[List[str]] = ["json"]
    output_format_default: ClassVar[str] = "json"

    def _export_output(self, path: Any, fmt: str) -> None:
        """Write the scores to ``path``, adding the extension."""
        if fmt != "json":
            raise ValueError(f"Unsupported output format {fmt!r}")
        import json

        path = Path(path).with_suffix(f".{fmt}")

        with open(path, "w") as handle:
            json.dump({"scores": self.scores}, handle)


def build_dispatch_input(
    inputs: Union[ProteinMPNNSampleInput, ProteinMPNNScoreInput],
    config: Union[ProteinMPNNSampleConfigBase, ProteinMPNNScoreConfigBase],
    operation: str,
) -> Dict[str, Any]:
    """Flatten an input and config pair into the dict a standalone receives.

    Everything crossing into a tool environment is plain JSON, so this keeps the
    two toolkits' dispatch payloads identical rather than letting them drift.

    Args:
        inputs: The tool input.
        config: The tool config.
        operation: ``"sample"`` or ``"score"``.

    Returns:
        A JSON-serializable payload.
    """
    payload: Dict[str, Any] = {
        "operation": operation,
        "structure": inputs.structure,
        "fixed_positions": inputs.fixed_positions,
        "chain_ids": inputs.chain_ids,
        "weights": config.weights,
        "device": config.device,
        "seed": config.seed,
        "verbose": config.verbose,
    }
    if operation == "sample":
        payload["prompt"] = inputs.prompt
        payload["num_sequences_to_generate"] = config.num_sequences_to_generate
        payload["temperature"] = config.temperature
        payload["excluded_amino_acids"] = config.excluded_amino_acids
    else:
        payload["sequences"] = inputs.sequences
    return payload
