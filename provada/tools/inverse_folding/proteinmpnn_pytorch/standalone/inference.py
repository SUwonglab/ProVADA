"""ProteinMPNN sampling and scoring on a PyTorch fork with prompt conditioning.

Runs inside this toolkit's environment, so it may import only the standard
library, its own requirements, and ``standalone_helpers``.

Every non-``_`` residue of the prompt overwrites the sequence parsed from the
structure, so all of it becomes conditioning context. This lets the model
condition on a supplied sequence rather than the one in the structure file,
which matters when the two have diverged. ``proteinmpnn_jax`` instead reads the
prompt at ``fixed_positions`` only.
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

from standalone_helpers import get_logger, move_model_to_device

logger = get_logger(__name__)

#: ``weights`` to the fork's constructor flags.
_WEIGHTS_CONFIG = {
    "original": {"use_soluble_model": False},
    "soluble": {"use_soluble_model": True},
}

#: Name the fork gives a structure passed as PDB text, from
#: ``protein_mpnn_utils.parse_PDB``. Per-protein arguments key off it.
_PROTEIN_NAME = "protein"

_MODEL: Optional["MPNNRunner"] = None


class MPNNRunner:
    """Holds one loaded ProteinMPNN for a single weight set."""

    def __init__(self, weights: str, device: str) -> None:
        """
        Args:
            weights: ``"original"`` or ``"soluble"``.
            device: Device to load onto.

        Note:
            The fork takes no seed at construction; it is a per-call argument on
            ``sample`` and ``score``.
        """
        from protein_mpnn import ProteinMPNN

        if weights not in _WEIGHTS_CONFIG:
            raise ValueError(
                f"proteinmpnn_pytorch: unknown weights {weights!r}; expected one of "
                f"{sorted(_WEIGHTS_CONFIG)}. AbMPNN is only available in the JAX "
                "toolkit, which loads it through ColabDesign."
            )
        self.weights = weights
        self.device = device
        self.model = ProteinMPNN(
            device=device,
            suppress_print=True,
            **_WEIGHTS_CONFIG[weights],
        )
        self._loaded = True

    def to_device(self, device: str) -> None:
        """Move the loaded weights between devices in place."""
        inner = getattr(self.model, "model", None)
        if inner is not None:
            move_model_to_device(inner, device)
        self.model.device = device
        self.device = device


def _get_model(weights: str, device: str) -> "MPNNRunner":
    """Return a cached runner, reloading only when the weight set changes."""
    global _MODEL
    if _MODEL is None or _MODEL.weights != weights:
        logger.info("Loading ProteinMPNN (%s weights)", weights)
        _MODEL = MPNNRunner(weights, device)
    elif _MODEL.device != device:
        _MODEL.to_device(device)
    return _MODEL


def _structure_to_pdb(structure: str) -> str:
    """Return PDB content, reading it from disk when given a path."""
    if "\n" not in structure and os.path.isfile(structure):
        with open(structure) as handle:
            return handle.read()
    return structure


def _prompt_to_chain_dict(
    prompt: Optional[str], chain_order: List[str]
) -> Optional[Dict[str, str]]:
    """Split a ``/``-joined prompt into the per-chain dict the fork expects.

    Args:
        prompt: Sequence with ``_`` at designed positions, chains split by ``/``.
        chain_order: Chain IDs in the order the prompt's segments appear.

    Returns:
        Chain ID to prompt segment, or ``None`` when there is no prompt.
    """
    if not prompt:
        return None
    segments = prompt.split("/")
    return {
        chain_id: segments[index]
        for index, chain_id in enumerate(chain_order)
        if index < len(segments)
    }


def _excluded_to_omit_string(excluded_amino_acids: Optional[List[str]]) -> str:
    """Build the fork's ``omit_AAs`` string.

    ``X`` is always omitted, matching the fork's own default, so a design never
    places an unknown residue.
    """
    omit = {"X"}
    for residue in excluded_amino_acids or []:
        omit.add(residue.upper())
    return "".join(sorted(omit))


def _nest_fixed_positions(
    fixed_positions: Optional[Dict[str, List[int]]],
) -> Optional[Dict[str, Dict[str, List[int]]]]:
    """Nest fixed positions under the structure's name, as the fork expects."""
    if not fixed_positions:
        return None
    return {_PROTEIN_NAME: dict(fixed_positions)}


def _chain_order(pdb: str, chain_ids: Optional[List[str]]) -> List[str]:
    """Chain IDs in the order they appear in the structure."""
    if chain_ids:
        return list(chain_ids)
    seen: List[str] = []
    for line in pdb.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            chain = line[21]
            if chain not in seen:
                seen.append(chain)
    return seen or ["A"]


def _first_protein(results: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the single structure's entry out of the fork's per-protein results."""
    if _PROTEIN_NAME in results:
        return results[_PROTEIN_NAME]
    if len(results) == 1:
        return next(iter(results.values()))
    raise ValueError(
        f"proteinmpnn_pytorch: expected one structure in results, got {sorted(results)}"
    )


def sample(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sample sequences for one structure."""
    seed = input_dict.get("seed")
    if seed is None:
        raise ValueError("proteinmpnn_pytorch: sample requires an explicit int seed")

    device = input_dict.get("device", "cuda")
    runner = _get_model(input_dict.get("weights", "original"), device)
    pdb = _structure_to_pdb(input_dict["structure"])
    chain_order = _chain_order(pdb, input_dict.get("chain_ids"))

    num_sequences = int(input_dict.get("num_sequences_to_generate", 10))
    results = runner.model.sample(
        pdb_path_or_str=pdb,
        prompt=_prompt_to_chain_dict(input_dict.get("prompt"), chain_order),
        num_seq_per_target=num_sequences,
        batch_size=num_sequences,
        # The fork takes the temperature as a string.
        sampling_temp=str(float(input_dict.get("temperature", 0.2))),
        seed=seed,
        fixed_positions_dict=_nest_fixed_positions(input_dict.get("fixed_positions")),
        omit_AAs=_excluded_to_omit_string(input_dict.get("excluded_amino_acids")),
    )

    entry = _first_protein(results)
    return {
        "sequences": [str(s) for s in entry["sequences"][:num_sequences]],
        "scores": [float(v) for v in entry["scores"][:num_sequences]],
    }


def score(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Score sequences against one structure.

    The fork reads supplied sequences from ``fasta_path`` only — its ``seq_str``
    argument is accepted and never read, and ``pdb_scores`` holds the score of
    the sequence parsed from the structure — so each sequence goes through a
    temporary FASTA.

    Each sequence is sent in its own call. The fork seeds once per call and then
    draws a fresh decoding order per sequence within it, so sending a batch
    would make a sequence's score depend on its position in the list. One call
    each keeps the scores comparable across requests.
    """
    seed = input_dict.get("seed")
    if seed is None:
        raise ValueError("proteinmpnn_pytorch: score requires an explicit int seed")

    device = input_dict.get("device", "cuda")
    runner = _get_model(input_dict.get("weights", "original"), device)
    pdb = _structure_to_pdb(input_dict["structure"])
    sequences = list(input_dict.get("sequences") or [])
    nested_fixed = _nest_fixed_positions(input_dict.get("fixed_positions"))

    if not sequences:
        # No sequence supplied scores the one parsed from the structure, which
        # is exactly what the fork returns as pdb_scores.
        results = runner.model.score(
            pdb_path_or_str=pdb, seed=seed, fixed_positions_dict=nested_fixed
        )
        entry = _first_protein(results)
        return {"scores": [float(entry["pdb_scores"]["mean_score"])]}

    scores: List[float] = []

    with tempfile.TemporaryDirectory() as workdir:
        fasta_path = os.path.join(workdir, "sequence.fasta")
        for index, sequence in enumerate(sequences):
            # The fork strips '/' when parsing, so a multi-chain sequence is
            # scored as the concatenation of its chains, which is how the
            # structure's own residues are laid out too.
            flat = sequence.replace("/", "")
            with open(fasta_path, "w") as handle:
                handle.write(f">sequence_{index}\n{flat}\n")

            results = runner.model.score(
                pdb_path_or_str=pdb,
                fasta_path=fasta_path,
                seed=seed,
                fixed_positions_dict=nested_fixed,
            )
            entry = _first_protein(results)

            # A sequence shorter than the structure has only its prefix
            # substituted, silently scoring the structure's own residues for
            # the remainder.
            structure_length = len(entry["pdb_scores"]["sequence"])
            if len(flat) != structure_length:
                raise ValueError(
                    f"proteinmpnn_pytorch: sequence {index} has {len(flat)} "
                    f"residues but the structure's scored chains have "
                    f"{structure_length}; every chain must be supplied, in the "
                    "order the structure declares them"
                )
            scores.append(float(entry["fasta_scores"][0]["mean_score"]))

    return {"scores": scores}


def to_device(device: str) -> Dict[str, Any]:
    """Device protocol: move the loaded model if there is one."""
    global _MODEL
    if _MODEL is not None and _MODEL._loaded:
        _MODEL.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> Dict[str, Any]:
    """Report this worker's PyTorch memory use."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats()


def dispatch(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Route a request to the requested operation."""
    operation = input_dict.get("operation", "sample")
    if operation == "sample":
        return sample(input_dict)
    if operation == "score":
        return score(input_dict)
    raise ValueError(f"proteinmpnn_pytorch: unknown operation {operation!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "proteinmpnn_pytorch: usage: python inference.py <input_json> <output_json>"
        )
    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    result = dispatch(payload)
    from standalone_helpers import serialize_output

    with open(sys.argv[2], "w") as handle:
        json.dump(serialize_output(result), handle)
