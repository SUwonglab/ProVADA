"""ProteinMPNN sampling and scoring on ColabDesign's JAX implementation.

Runs inside this toolkit's environment, so it may import only the standard
library, its own requirements, and ``standalone_helpers``.

The prompt is applied at the fixed positions only: each one's sequence input is
set to the prompt's residue and that residue is given a large bias, pinning it.
Positions outside ``fixed_positions`` are designed from the backbone alone, so a
residue the prompt specifies there has no effect. ``proteinmpnn_pytorch`` treats
those positions as conditioning context instead.

The PRNG key is derived from ``seed``, so repeated calls with the same inputs
return the same sequences.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from standalone_helpers import get_logger, set_jax_seed

logger = get_logger(__name__)

#: Amino-acid order ProteinMPNN's logits use (BLOSUM ordering).
ALPHABET_BLOSUM = "ARNDCQEGHILKMFPSTWYV"

#: Bias magnitude that pins a position to one residue. Large enough to dominate
#: the logits without overflowing in float32.
_PIN_BIAS = 1e7

#: Weight set to ColabDesign's (model_name, weights) pair. AbMPNN loads by name
#: from ColabDesign's weights directory, where the borrowed proto-tools
#: environment's setup.sh places the checkpoint. Mirrors the mapping in
#: proto-tools' own proteinmpnn toolkit.
_WEIGHTS_CONFIG = {
    "original": ("v_48_020", "original"),
    "soluble": ("v_48_020", "soluble"),
    "antibody": ("abmpnn", "original"),
}

_MODEL: Optional["MPNNRunner"] = None


class MPNNRunner:
    """Holds one loaded ColabDesign MPNN model for a single weight set."""

    def __init__(self, weights: str, seed: int) -> None:
        """
        Args:
            weights: ``"original"``, ``"soluble"`` or ``"antibody"``.
            seed: Seed for the model's own RNG.
        """
        from colabdesign.mpnn import mk_mpnn_model

        if weights not in _WEIGHTS_CONFIG:
            raise ValueError(
                f"proteinmpnn_jax: unknown weights {weights!r}; "
                f"expected one of {sorted(_WEIGHTS_CONFIG)}"
            )
        model_name, colabdesign_weights = _WEIGHTS_CONFIG[weights]
        self.weights = weights
        self.model = mk_mpnn_model(
            model_name=model_name, weights=colabdesign_weights, seed=seed
        )
        self._loaded = True

    def to_device(self, device: str) -> None:
        """No-op: a JAX runtime is pinned to its visible devices at load."""
        return None


def _get_model(weights: str, seed: int) -> MPNNRunner:
    """Return a cached runner, reloading only when the weight set changes."""
    global _MODEL
    if _MODEL is None or _MODEL.weights != weights:
        logger.info("Loading ProteinMPNN (%s weights)", weights)
        _MODEL = MPNNRunner(weights, seed)
    return _MODEL


def _structure_to_pdb(structure: str) -> str:
    """Return PDB content, reading it from disk when given a path."""
    if "\n" not in structure and os.path.isfile(structure):
        with open(structure) as handle:
            return handle.read()
    return structure


def _prep(model: Any, pdb: str, chain_ids, excluded_amino_acids) -> None:
    """Parse the structure into the model's inputs.

    ``X`` is dropped from the exclusion list: ColabDesign turns ``rm_aa`` into a
    bias over the 20 canonical residues, which has no ``X`` column, so asking
    for it raises. The model cannot emit ``X`` anyway.
    """
    excluded = [aa for aa in (excluded_amino_acids or []) if aa.upper() != "X"]
    rm_aa = ",".join(excluded) if excluded else None
    model.prep_inputs(
        pdb,
        chain=",".join(chain_ids) if chain_ids else None,
        rm_aa=rm_aa,
        ignore_missing=False,
    )


def _pin_prompt_at_fixed_positions(
    model: Any,
    prompt: Optional[str],
    fixed_positions: Optional[Dict[str, List[int]]],
    chain_order: List[str],
) -> int:
    """Pin each fixed position to the prompt's residue there.

    Writes the residue into the model's sequence input and biases its logit, so
    the position is held while the rest of the chain is designed. Positions are
    1-indexed within their own chain, and ``prompt`` is split on ``/`` in the
    same chain order the structure was parsed in.

    Returns:
        How many positions were pinned.
    """
    if not prompt or not fixed_positions:
        return 0

    sequence_input = model._inputs["S"]
    bias = model._inputs.get("bias", None)
    aa_to_index = {aa: i for i, aa in enumerate(ALPHABET_BLOSUM)}
    lengths = model._lengths
    segments = prompt.split("/")

    pinned = 0
    offset = 0
    for chain_index, chain_id in enumerate(chain_order):
        chain_length = lengths[chain_index]
        positions = fixed_positions.get(chain_id)
        segment = segments[chain_index] if chain_index < len(segments) else ""

        for position in positions or []:
            if position < 1 or position > chain_length or position > len(segment):
                continue
            residue = segment[position - 1]
            residue_index = aa_to_index.get(residue)
            if residue_index is None:
                continue

            target = offset + (position - 1)
            if target < len(sequence_input):
                sequence_input[target] = residue_index
            if bias is not None and target < bias.shape[0]:
                bias[target, :] = 0.0
                if residue_index < bias.shape[1]:
                    bias[target, residue_index] = _PIN_BIAS
            pinned += 1

        offset += chain_length
    return pinned


def _chain_order(model: Any, chain_ids) -> List[str]:
    """Chain IDs in the order the model parsed them."""
    if chain_ids:
        return list(chain_ids)
    parsed = getattr(model, "_chains", None)
    if parsed:
        return list(parsed)
    # One unnamed chain is the common case for a monomer.
    return ["A"]


def sample(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sample sequences for one structure."""
    seed = input_dict.get("seed")
    if seed is None:
        raise ValueError("proteinmpnn_jax: sample requires an explicit int seed")
    key = set_jax_seed(seed)

    model = _get_model(input_dict.get("weights", "original"), seed).model
    pdb = _structure_to_pdb(input_dict["structure"])
    chain_ids = input_dict.get("chain_ids")

    _prep(model, pdb, chain_ids, input_dict.get("excluded_amino_acids"))
    pinned = _pin_prompt_at_fixed_positions(
        model,
        input_dict.get("prompt"),
        input_dict.get("fixed_positions"),
        _chain_order(model, chain_ids),
    )
    if pinned:
        logger.info("Pinned %d position(s) from the prompt", pinned)

    num_sequences = int(input_dict.get("num_sequences_to_generate", 10))
    results = model.sample(
        batch=num_sequences,
        temperature=float(input_dict.get("temperature", 0.2)),
        key=key,
    )
    return {
        "sequences": [str(s) for s in results["seq"][:num_sequences]],
        "scores": [float(v) for v in results["score"][:num_sequences]],
    }


def score(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Score sequences against one structure."""
    seed = input_dict.get("seed")
    if seed is None:
        raise ValueError("proteinmpnn_jax: score requires an explicit int seed")
    key = set_jax_seed(seed)

    model = _get_model(input_dict.get("weights", "original"), seed).model
    pdb = _structure_to_pdb(input_dict["structure"])
    chain_ids = input_dict.get("chain_ids")
    _prep(model, pdb, chain_ids, None)

    sequences = input_dict.get("sequences")
    if not sequences:
        # No sequence supplied scores the one parsed from the structure.
        return {"scores": [float(model.score(key=key)["score"])]}

    # One key for every sequence, so a score does not depend on where the
    # sequence sat in the request.
    return {
        "scores": [float(model.score(seq=sequence, key=key)["score"]) for sequence in sequences]
    }


def to_device(device: str) -> Dict[str, Any]:
    """Device protocol. JAX pins its devices at load, so nothing moves."""
    return {
        "success": True,
        "device": device,
        "note": "JAX tool, devices fixed at load",
    }


def get_memory_stats() -> Dict[str, Any]:
    """Report this worker's JAX memory use."""
    from standalone_helpers import get_jax_memory_stats

    return get_jax_memory_stats()


def dispatch(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Route a request to the requested operation."""
    operation = input_dict.get("operation", "sample")
    if operation == "sample":
        return sample(input_dict)
    if operation == "score":
        return score(input_dict)
    raise ValueError(f"proteinmpnn_jax: unknown operation {operation!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "proteinmpnn_jax: usage: python inference.py <input_json> <output_json>"
        )
    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    result = dispatch(payload)
    from standalone_helpers import serialize_output

    with open(sys.argv[2], "w") as handle:
        json.dump(serialize_output(result), handle)
