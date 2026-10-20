"""
generator.py
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Any, Dict
from pathlib import Path
from provada.sequences.mutate import sample_amino_acid
from provada.utils.setup import seed_everything
import pandas as pd
from provada.utils.registry import GENERATOR_REGISTRY
from tqdm import tqdm
from provada.utils.env import resolve_device
from provada.utils.structure import ProteinStructure
from provada.utils.log import get_logger

logger = get_logger(__name__)


def _parse_fixed_positions(prompt: str, structure: Union[ProteinStructure, List[ProteinStructure]]) -> Dict[str, List[int]]:
    """
    Parses a prompt string into a dictionary of fixed positions for each chain.

    Args:
        prompt: The design prompt string, potentially containing '/' sequence separators.
        structure: The protein structure (or list of structures) to align with.

    Returns:
        A dictionary mapping chain IDs to lists of 1-based fixed position indices.

    Raises:
        ValueError: If prompt chain count doesn't match structure chain count.
    """
    if isinstance(structure, list):
        # validation: ensure all structures have same chains? assuming yes for ensemble
        structure_to_use = structure[0]
    else:
        structure_to_use = structure

    chain_dict = structure_to_use.get_chain_sequences()
    chain_ids = list(chain_dict)
    
    prompt_segments = prompt.split("/")
    
    if len(prompt_segments) != len(chain_ids):
         raise ValueError(f"Number of chains in prompt ({len(prompt_segments)}) does not match number of chains in structure ({len(chain_ids)}). Prompt: '{prompt}', Chains: {chain_ids}")

    position_map = structure_to_use.get_residue_position_map() if hasattr(structure_to_use, "get_residue_position_map") else None
    fixed_positions = {}
    for chain_id, segment in zip(chain_ids, prompt_segments):
        missing = getattr(structure_to_use, "missing_residues_triplets", {}).get(chain_id, [])
        resolved_entries = position_map.get(chain_id, []) if position_map is not None else []
        resolved = {position for _, position in resolved_entries}
        # Gemmi's display sequence can contain '-' gap placeholders; count
        # actual residues and explicit missing-residue records instead.
        resolved_count = len(resolved_entries) if position_map is not None else len(chain_dict[chain_id])
        expected = resolved_count + len({position for position, _, _ in missing if position not in resolved})
        if len(segment) != expected:
            raise ValueError(f"Prompt chain {chain_id} has {len(segment)} residues; expected {expected}")
        indices = [i + 1 for i, char in enumerate(segment) if char != "_"]
        if indices:
            fixed_positions[chain_id] = indices
            
    return fixed_positions


def is_single_structure(
    structure_or_structures: Union[
        str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
    ],
) -> bool:
    """
    Returns True when one structure was given to use for every sequence, rather
    than one structure per sequence.
    """
    return isinstance(structure_or_structures, (str, Path, ProteinStructure))


def as_structures(
    structure_or_structures: Union[
        str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
    ],
    count: int,
) -> List[ProteinStructure]:
    """
    Returns one ProteinStructure per sequence.

    A single structure is broadcast across all ``count`` sequences; a list is
    taken as a 1-to-1 mapping and used as given.
    """
    if is_single_structure(structure_or_structures):
        return [ProteinStructure(structure_or_structures)] * count
    return [ProteinStructure(structure) for structure in structure_or_structures]


def as_ensemble(
    structure_or_structures: Union[
        str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
    ],
) -> List[ProteinStructure]:
    """
    Returns a conformer ensemble, the primary conformer first.

    A single structure is an ensemble of one, which is how a generator that
    conditions on an ensemble also handles being given one structure.
    """
    if is_single_structure(structure_or_structures):
        return [ProteinStructure(structure_or_structures)]

    ensemble = [ProteinStructure(structure) for structure in structure_or_structures]
    if not ensemble:
        raise ValueError("An ensemble needs at least one structure")
    return ensemble


def rows_with_generations(
    row: pd.Series, generations: List[Dict[str, Any]]
) -> List[pd.Series]:
    """
    Carries one prompt's row metadata onto every design generated from it.

    A prompt can produce several designs, so its row is copied once per design
    and the design's own fields are written over it.
    """
    rows = []
    for generation in generations:
        # Casting to object first keeps the row from coercing added fields, and
        # drops the index so the row does not carry its old position.
        new_row = row.astype(object)
        for key, value in generation.items():
            new_row[key] = value
        new_row.name = None
        rows.append(new_row)
    return rows


def get_generator(name: str, **kwargs):
    """
    Get a generator instance from the registry.
    """
    return GENERATOR_REGISTRY.get_instance(name, **kwargs)


class Generator(ABC):
    """
    Generator is an abstract class for all generators.
    """

    def score(self, sequences: Union[List[str], pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Scores a list of sequences (or dataframe).
        Returns a dataframe with scores.
        """
        # Default implementation returns empty dataframe
        # Subclasses should override this if they support scoring
        length = len(sequences)
        return pd.DataFrame(index=range(length))

    def generate(self, prompts: Union[List[str], pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Generate one sequence per prompt, preserving metadata by row position.

        Returned rows have a fresh index, and generated fields replace input
        fields with the same names. Generators with several designs per prompt
        override this wrapper to expand the corresponding metadata explicitly.
        """
        if isinstance(prompts, list):
            prompts = pd.DataFrame({"prompt": prompts})
        else:
            prompts = prompts.reset_index(drop=True)
            # Ensure that the "prompt" column exists
            if "prompt" not in prompts.columns:
                raise ValueError("Input dataframe to generator must contain a 'prompt' column")
        if not prompts.columns.is_unique:
            raise ValueError("Input dataframe to generator must have unique column names")
        if len(prompts) == 0:
            return prompts.assign(sequence=pd.Series(dtype=object))

        # Call the generator
        try:
            generation_output = pd.DataFrame(
                self._generate(prompts["prompt"].tolist(), **kwargs)
            ).reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error in _generate function of {self.__class__.__name__}: {e}")
            raise e

        if len(generation_output) != len(prompts):
            raise ValueError(
                f"Generator returned {len(generation_output)} rows for {len(prompts)} prompts; "
                "the base wrapper requires one output row per prompt"
            )
        if not generation_output.columns.is_unique:
            raise ValueError("Generator output must have unique column names")
        if "sequence" not in generation_output:
            raise ValueError("Generator output must contain a 'sequence' column")

        # Preserve input-only metadata; new sequences and scores take precedence.
        metadata = prompts.drop(columns=generation_output.columns, errors="ignore")
        return pd.concat([metadata, generation_output], axis=1)

    @abstractmethod
    def _generate(self, prompts: list[str], **kwargs) -> list[dict]:
        """
        Use the generator to generate sequences. Assumes input sequences have
        masked positions represented by '_'. This method should be defined in
        the child class.

        Args:
            prompts: list of sequences to generate
            **kwargs: additional arguments to pass to the generation method

        Returns:
            List of dictionaries with the a "sequence" key and associated sequence
            metadata under other keys
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_extra_kwargs(self, base_variant):
        """
        Returns a dictionary of extra keyword arguments needed for generation.
        Subclasses can override this method to specify additional kwargs required
        for their generate method.

        Args:
            base_variant: The base variant object that may contain information
                needed for generation (e.g., structure for MPNN).

        Returns:
            A dictionary of extra keyword arguments.
        """
        return {}


class MaskedModelGenerator(Generator):
    """Base for generators backed by a proto-tools masked-model sampling tool.

    The model runs in proto-tools' own environment, so nothing here imports a
    model stack or moves weights between devices. ``device`` is forwarded into
    the tool config and proto-tools' DeviceManager decides placement.
    """

    def __init__(self, seed: int = 42, device: Optional[str] = None, **kwargs: Any):
        """
        Args:
            seed: Seed for this generator's draws.
            device: Device string forwarded to the sampling tool. ``None``
                leaves the tool's own default in place.
        """
        self.seed = seed
        self.device = device
        self.seed_rng = np.random.default_rng(seed)
        seed_everything(seed)

    def _dispatch(
        self,
        run_tool: Any,
        input_cls: Any,
        config_cls: Any,
        prompts: List[str],
        device: Optional[str],
        **config_kwargs: Any,
    ) -> list[dict]:
        """Sample one batch of prompts and pair each result with its prompt.

        proto-tools returns one result per input in input order, so the pairing
        is positional and asserted by ``strict=True``.
        """
        config_kwargs["device"] = resolve_device(device or self.device)

        # Only send an exclusion list when there is something to exclude:
        # the tool configs reject unknown fields, so an older proto-tools
        # without `excluded_amino_acids` still serves calls that do not use it.
        excluded = config_kwargs.pop("excluded_amino_acids", None)
        if excluded:
            config_kwargs["excluded_amino_acids"] = excluded

        outputs = run_tool(
            input_cls(sequences=prompts),
            config_cls(seed=int(self.seed_rng.integers(0, 2**31 - 1)), **config_kwargs),
        )
        return [
            {"sequence": result.sequence, "prompt": prompt}
            for result, prompt in zip(outputs.results, prompts, strict=True)
        ]


@GENERATOR_REGISTRY.register("esm3")
class ESM3Generator(MaskedModelGenerator):
    """Fills masked positions with ESM3, via proto-tools' ``esm3-sample``."""

    def _generate(
        self,
        prompts: List[str],
        max_batch_size: int = 20,
        device: Optional[str] = None,
        schedule: str = "cosine",
        strategy: str = "entropy",
        num_steps: int = 16,
        temperature: float = 0.8,
        temperature_annealing: bool = True,
        top_p: float = 0.95,
        excluded_amino_acids: Optional[List[str]] = None,
        sampling_method: str = "iterative_refinement",
        model_checkpoint: str = "esm3_sm_open_v1",
        **kwargs: Any,
    ) -> list[dict]:
        """
        Generate sequences with ESM3.

        Args:
            prompts: Sequences with ``_`` at the positions to fill in.
            max_batch_size: Sequences per forward pass.
            device: Device for this call; defaults to the generator's.
            schedule: Unmasking schedule, ``"cosine"`` or ``"linear"``. Iterative only.
            strategy: Which positions to commit each round, ``"random"`` or
                ``"entropy"``. Iterative only.
            num_steps: Iterative decoding steps. Iterative only.
            temperature: Sampling temperature.
            temperature_annealing: Anneal the temperature across steps.
            top_p: Nucleus-sampling threshold.
            excluded_amino_acids: Single-letter codes the model may not
                place at masked positions, e.g. ``["C"]``.
            sampling_method: ``"iterative_refinement"`` reproduces the
                multi-step decoding;
                ``"single_pass"`` fills every position in one forward pass and
                ignores the schedule, strategy and step count.
            model_checkpoint: ESM3 weights to load.

        Returns:
            One dict per prompt with ``sequence`` and ``prompt`` keys.
        """
        from proto_tools.tools import ESM3SampleConfig, ESM3SampleInput, run_esm3_sample

        return self._dispatch(
            run_esm3_sample,
            ESM3SampleInput,
            ESM3SampleConfig,
            prompts,
            device,
            batch_size=min(max_batch_size, len(prompts)),
            sampling_method=sampling_method,
            model_checkpoint=model_checkpoint,
            schedule=schedule,
            strategy=strategy,
            num_steps=num_steps,
            temperature=temperature,
            temperature_annealing=temperature_annealing,
            top_p=top_p,
            excluded_amino_acids=excluded_amino_acids,
        )


@GENERATOR_REGISTRY.register("esm2")
class ESM2Generator(MaskedModelGenerator):
    """Fills masked positions with ESM2, via proto-tools' ``esm2-sample``."""

    def _generate(
        self,
        prompts: List[str],
        max_batch_size: int = 20,
        device: Optional[str] = None,
        schedule: str = "cosine",
        strategy: str = "entropy",
        num_steps: int = 16,
        temperature: float = 0.8,
        temperature_annealing: bool = True,
        top_p: float = 0.95,
        excluded_amino_acids: Optional[List[str]] = None,
        sampling_method: str = "iterative_refinement",
        model_checkpoint: str = "esm2_t33_650M_UR50D",
        **kwargs: Any,
    ) -> list[dict]:
        """
        Generate sequences with ESM2.

        Takes the same arguments as :class:`ESM3Generator`, since proto-tools
        gives both tools one masked-model sampling surface. See that docstring
        for the per-argument detail.

        Returns:
            One dict per prompt with ``sequence`` and ``prompt`` keys.
        """
        from proto_tools.tools import ESM2SampleConfig, ESM2SampleInput, run_esm2_sample

        return self._dispatch(
            run_esm2_sample,
            ESM2SampleInput,
            ESM2SampleConfig,
            prompts,
            device,
            batch_size=min(max_batch_size, len(prompts)),
            sampling_method=sampling_method,
            model_checkpoint=model_checkpoint,
            schedule=schedule,
            strategy=strategy,
            num_steps=num_steps,
            temperature=temperature,
            temperature_annealing=temperature_annealing,
            top_p=top_p,
            excluded_amino_acids=excluded_amino_acids,
        )


@GENERATOR_REGISTRY.register("mpnn")
class MPNNGenerator(Generator):
    """Designs sequences with ProteinMPNN, through one of ProVADA's toolkits.

    The model runs in the toolkit's own isolated environment, so nothing here
    imports ProteinMPNN or moves weights between devices: ``device`` is
    forwarded into the tool config and proto-tools' DeviceManager decides
    placement.

    Which implementation and weight set to use is part of the generator's
    identity, not a parameter: each registry entry sets ``_TOOLKIT`` and
    ``_WEIGHTS``. See ``provada/tools/inverse_folding/shared_data_models.py``
    for how the two implementations differ.
    """

    #: ``"jax"`` or ``"pytorch"``; selects which toolkit this generator calls.
    _TOOLKIT: str = "jax"

    #: ``"original"``, ``"soluble"`` or ``"antibody"``.
    _WEIGHTS: str = "original"

    #: Repeated copies of an ancestor share a mask so their completions can be
    #: generated together in one ProteinMPNN batch.
    batch_shared_prompts: bool = True

    def __init__(self, seed: int = 42, device: Optional[str] = None, **kwargs):
        """
        Args:
            seed: Seed forwarded to the tool, which requires one.
            device: Device string forwarded to the tool. ``None`` leaves the
                tool's own default in place.
        """
        self.seed = seed
        self.device = device
        self.seed_rng = np.random.default_rng(seed)

    def _tools(self):
        """Return this generator's ``(sample, score)`` tool entry points.

        Imported lazily and by flavor so a run that only uses one toolkit never
        touches the other.
        """
        import provada.tools  # noqa: F401 - registers ProVADA's toolkits

        if self._TOOLKIT == "jax":
            from provada.tools.inverse_folding.proteinmpnn_jax import (
                ProteinMPNNJaxSampleConfig,
                ProteinMPNNJaxScoreConfig,
                run_proteinmpnn_jax_sample,
                run_proteinmpnn_jax_score,
            )

            return (
                (run_proteinmpnn_jax_sample, ProteinMPNNJaxSampleConfig),
                (run_proteinmpnn_jax_score, ProteinMPNNJaxScoreConfig),
            )

        if self._TOOLKIT == "pytorch":
            from provada.tools.inverse_folding.proteinmpnn_pytorch import (
                ProteinMPNNPytorchSampleConfig,
                ProteinMPNNPytorchScoreConfig,
                run_proteinmpnn_pytorch_sample,
                run_proteinmpnn_pytorch_score,
            )

            return (
                (run_proteinmpnn_pytorch_sample, ProteinMPNNPytorchSampleConfig),
                (run_proteinmpnn_pytorch_score, ProteinMPNNPytorchScoreConfig),
            )

        raise ValueError(
            f"{type(self).__name__}: unknown toolkit {self._TOOLKIT!r}; "
            "expected 'jax' or 'pytorch'"
        )

    def _tool_kwargs(self, device: Optional[str]) -> Dict[str, Any]:
        """Config arguments every call to either tool shares."""
        kwargs: Dict[str, Any] = {"seed": self.seed, "weights": self._WEIGHTS}
        kwargs["device"] = resolve_device(device or self.device)
        return kwargs

    def get_extra_kwargs(self, base_variant):
        """
        MPNNGenerator requires a protein structure for generation.
        """
        return {"structure_or_structures": base_variant.structure}

    def score(
        self,
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        sequences: Union[List[str], pd.DataFrame, str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculates MPNN score (negative log likelihood) for the given sequences.
        Scoring does not necessarily need a sequence, it can directly score a structure with sequence.

        Every sequence is scored under the same decoding order, so a score
        does not depend on where its sequence sat in the request.
        """
        from provada.tools.inverse_folding.shared_data_models import (
            ProteinMPNNScoreInput,
        )

        if isinstance(sequences, pd.DataFrame):
            if "sequence" not in sequences.columns:
                 raise ValueError("Input dataframe must have 'sequence' column")
            seqs = sequences["sequence"].tolist()
        elif isinstance(sequences, str):
            seqs = [sequences]
        else:
            seqs = sequences

        (_, _), (run_score, score_config) = self._tools()
        config = score_config(**self._tool_kwargs(device))

        if is_single_structure(structure_or_structures):
            # Fast path: one structure, every sequence in one dispatch. `seqs`
            # of None scores the structure's own sequence, which is what the
            # base variant's score goes through.
            structure = ProteinStructure(structure_or_structures)
            mpnn_scores = run_score(
                ProteinMPNNScoreInput(
                    structure=structure.structure_pdb,
                    sequences=seqs,
                    chain_ids=structure.get_chain_ids(),
                ),
                config,
            ).scores
        else:
            structures = [ProteinStructure(s) for s in structure_or_structures]
            if seqs is None or len(structures) != len(seqs):
                 raise ValueError("Number of structures must match sequences if list of structures provided.")

            mpnn_scores = []
            for seq, structure in zip(seqs, structures):
                mpnn_scores.extend(
                    run_score(
                        ProteinMPNNScoreInput(
                            structure=structure.structure_pdb,
                            sequences=[seq],
                            chain_ids=structure.get_chain_ids(),
                        ),
                        config,
                    ).scores
                )

        return pd.DataFrame({"mpnn_score": mpnn_scores})

    def generate(
        self,
        prompts: Union[List[str], pd.DataFrame],
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        generations_per_sequence: int = 10,
        **kwargs,
    ):
        """
        Batch all requested offspring for each shared prompt and backbone.

        Each input row receives generations_per_sequence distinct draws from
        its group's batch, with its own parent metadata. Output rows retain
        input order even when matching prompts were interleaved.
        """

        # Convert list to DataFrame if needed
        if isinstance(prompts, list):
            prompts = pd.DataFrame({"prompt": prompts})

        # Ensure that the "prompt" column exists
        if "prompt" not in prompts.columns:
            raise ValueError("Input dataframe to generator must contain a 'prompt' column")

        if generations_per_sequence < 1 or int(generations_per_sequence) != generations_per_sequence:
            raise ValueError("generations_per_sequence must be a positive integer")
        generations_per_sequence = int(generations_per_sequence)
        structures = as_structures(structure_or_structures, len(prompts))
        if len(structures) != len(prompts):
            raise ValueError("Number of structures must match number of prompts")
        if len(prompts) == 0:
            return prompts.copy().assign(sequence=pd.Series(dtype=object), mpnn_score=pd.Series(dtype=float))

        # A matching mask alone is insufficient: the unmasked context, backbone
        # and chain order must also agree. All tool settings are shared by this
        # generate call. Cache structure serialization for broadcast backbones.
        structure_keys = {}
        groups = {}
        for position, (prompt, structure) in enumerate(zip(prompts["prompt"], structures, strict=True)):
            identity = id(structure)
            if identity not in structure_keys:
                structure_keys[identity] = (structure.structure_pdb, tuple(structure.get_chain_ids()))
            key = (prompt, structure_keys[identity])
            groups.setdefault(key, []).append(position)

        rows_by_position = [[] for _ in range(len(prompts))]
        for positions in tqdm(groups.values(), desc="Generating sequences with MPNN", total=len(groups), unit="batches"):
            first = positions[0]
            requested = len(positions) * generations_per_sequence
            generations = self._generate(
                prompts=[prompts.iloc[first]["prompt"]],
                structure_or_structures=structures[first],
                generations_per_sequence=requested,
                **kwargs,
            )
            if len(generations) != requested:
                raise ValueError(
                    f"ProteinMPNN returned {len(generations)} designs for a batch requesting {requested}"
                )
            for offset, position in enumerate(positions):
                start = offset * generations_per_sequence
                rows_by_position[position] = rows_with_generations(
                    prompts.iloc[position], generations[start:start + generations_per_sequence]
                )

        return pd.DataFrame([row for rows in rows_by_position for row in rows])

    def _generate(
        self,
        prompts: List[str],
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        generations_per_sequence: int = 10,
        temperature: float = 0.8,
        excluded_amino_acids: Optional[List[str]] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Design sequences for each prompt.

        Args:
            prompts: Sequences with ``_`` at the positions to design, chains
                split by ``/``.
            structure_or_structures: One structure for every prompt, or one per
                prompt.
            generations_per_sequence: Designs to sample per prompt.
            temperature: Sampling temperature.
            excluded_amino_acids: Single-letter codes the model may not place at
                designed positions, e.g. ``["C"]``.
            device: Device for this call; defaults to the generator's.

        Returns:
            One dict per design with ``sequence``, ``prompt`` and
            ``mpnn_score`` keys.
        """
        from provada.tools.inverse_folding.shared_data_models import (
            ProteinMPNNSampleInput,
        )

        structures = as_structures(structure_or_structures, len(prompts))

        (run_sample, sample_config), (_, _) = self._tools()
        tool_kwargs = self._tool_kwargs(device)

        generations = []
        for prompt, structure in zip(prompts, structures):
            # Scoring keeps a fixed seed; each proposal request advances this
            # generator's reproducible stream instead of replaying its batch.
            tool_kwargs["seed"] = int(self.seed_rng.integers(0, 2**31 - 1))
            config = sample_config(
                num_sequences_to_generate=generations_per_sequence,
                temperature=temperature,
                excluded_amino_acids=excluded_amino_acids,
                **tool_kwargs,
            )
            fixed_positions = _parse_fixed_positions(prompt, structure)

            # chain_ids in the structure's own order, which is the order the
            # prompt's '/'-separated segments are read in.
            incoming = run_sample(
                ProteinMPNNSampleInput(
                    structure=structure.structure_pdb,
                    prompt=prompt,
                    fixed_positions=fixed_positions,
                    chain_ids=structure.get_chain_ids(),
                ),
                config,
            )

            for sequence in incoming.sequences[:generations_per_sequence]:
                if len(sequence) != len(prompt) or any(
                    fixed != "_" and fixed != residue for fixed, residue in zip(prompt, sequence)
                ):
                    raise ValueError("ProteinMPNN output changed an unmasked residue or chain layout")

            generations.extend(
                {
                    "sequence": sequence,
                    "prompt": prompt,
                    "mpnn_score": score,
                }
                for sequence, score in zip(
                    incoming.sequences[:generations_per_sequence],
                    incoming.scores[:generations_per_sequence],
                )
            )

        return generations


@GENERATOR_REGISTRY.register("soluble_mpnn")
class SolubleMPNNGenerator(MPNNGenerator):
    """solubleMPNN, through the JAX toolkit."""

    _WEIGHTS = "soluble"


@GENERATOR_REGISTRY.register("abmpnn")
class AbMPNNGenerator(MPNNGenerator):
    """AbMPNN, through the JAX toolkit.

    Antibody weights load through ColabDesign, so there is no PyTorch
    equivalent of this generator.
    """

    _WEIGHTS = "antibody"


@GENERATOR_REGISTRY.register("mpnn_pytorch")
class MPNNPytorchGenerator(MPNNGenerator):
    """ProteinMPNN through the PyTorch toolkit.

    The same weights as :class:`MPNNGenerator`, but a different implementation:
    this one conditions on every non-masked residue of the prompt rather than
    only the fixed positions, so it does not produce the same designs.
    """

    _TOOLKIT = "pytorch"


@GENERATOR_REGISTRY.register("soluble_mpnn_pytorch")
class SolubleMPNNPytorchGenerator(MPNNPytorchGenerator):
    """solubleMPNN, through the PyTorch toolkit."""

    _WEIGHTS = "soluble"


@GENERATOR_REGISTRY.register("random")
class RandomGenerator(Generator):

    def __init__(self, seed: int = 42, device: Optional[str] = None, **kwargs: Any):
        """
        Set up the random generator. Takes a device for a uniform generator
        signature; sampling codons needs no model and ignores it.
        """

        self.seed_rng = np.random.default_rng(seed)
        self.device = device

    def _generate(
        self, prompts: List[str], codon_scheme: str = "NNN", **kwargs: Any
    ) -> list[dict]:
        """
        Infill masked positions of sequences with amino acids collected from sampling
        the specified codon scheme. NOTE that stop codons are disallowed.

        Args:
            prompts: list of sequences to infill the masked positions of. Masked
                positions are expected to be represented by '_'

        Returns:
            List of dictionaries with the a "sequence" key and associated sequence
            metadata under other keys
        """

        random_seed = self.seed_rng.integers(0, 1000000)
        random.seed(int(random_seed))

        prompt_lists = [list(seq) for seq in prompts]

        # For each sequence
        for prompt_list in prompt_lists:
            # For each position
            for i, char in enumerate(prompt_list):
                # If the position is masked, sample an amino acid
                if char == "_":
                    prompt_list[i] = sample_amino_acid(codon_scheme, allow_stop=False)

        return [
            {"sequence": "".join(prompt_list), "prompt": prompt}
            for prompt_list, prompt in zip(prompt_lists, prompts)
        ]

# ============================================================================
# Caliby
# ============================================================================


def chain_numbering(
    structure: ProteinStructure, proto_structure: Any
) -> Dict[str, Tuple[List[int], Dict[int, str]]]:
    """
    Reads the residue numbering of every chain, and which of its residues the
    model will not see.

    A ProVADA sequence covers a whole chain, while Caliby is only given the
    residues it can design, so prompt positions and designed positions have to
    be translated through the chain's numbering. A residue goes unseen either
    because the structure never resolved it, which ``REMARK 465`` records, or
    because it has no alpha carbon and was filtered out of the structure the
    model is given.

    Args:
        structure: The structure as ProVADA holds it, which defines the full
            chain and names the residues the coordinates omit.
        proto_structure: The structure as handed to the tool, already filtered.
            Its numbering is what resolves a 1-indexed position to a residue
            number, so reading it here keeps the two sides in agreement.

    Returns:
        Per chain, the full residue numbering in ascending order, and the
        one-letter code of every unseen residue by residue number.
    """
    position_map = structure.get_residue_position_map()

    numbering: Dict[str, Tuple[List[int], Dict[int, str]]] = {}
    model_chains = set(proto_structure.get_chain_ids())
    if model_chains != set(position_map):
        raise ValueError("Structure and converted structure must contain the same chains")
    for chain_id in position_map:
        residues = {number: residue for residue, number in position_map.get(chain_id, [])}
        # Coordinates win over a REMARK 465 record naming the same residue, so a
        # residue recorded as missing that has coordinates is not counted twice.
        residues.update(
            {
                number: residue
                for number, _icode, residue in structure.missing_residues_triplets.get(
                    chain_id, []
                )
                if number not in residues
            }
        )

        full_numbering = sorted(residues)
        designed = set(proto_structure.get_chain_positions(chain_id))
        unresolved = {
            number: residues[number]
            for number in full_numbering
            if number not in designed
        }
        numbering[chain_id] = (full_numbering, unresolved)

    return numbering


def prompt_pins(
    prompt: str, numbering: Dict[str, Tuple[List[int], Dict[int, str]]]
) -> Dict[str, Dict[int, str]]:
    """
    Collects the residue identities a prompt holds fixed.

    Every position the prompt does not mask is pinned to the residue the prompt
    carries there, which need not be the residue the structure carries: that is
    how a design keeps the variant it is being built from rather than reverting
    to the backbone's own sequence. Unresolved positions are left out, since
    Caliby never sees them.

    Args:
        prompt: One character per residue of the full sequence, ``_`` where the
            generator may design, chains split by ``/`` in structure order.
        numbering: Per-chain numbering, from :func:`chain_numbering`.

    Returns:
        Per chain, the 1-indexed position among that chain's resolved residues
        mapped to the residue to place there. Chains with nothing pinned are
        left out.

    Raises:
        ValueError: If the prompt's chain count, or any chain's length, does not
            match the structure.
    """
    chain_ids = list(numbering)
    segments = prompt.split("/")
    if len(segments) != len(chain_ids):
        raise ValueError(
            f"Number of chains in prompt ({len(segments)}) does not match number of "
            f"chains in structure ({len(chain_ids)}). Prompt: '{prompt}', "
            f"Chains: {chain_ids}"
        )

    pins: Dict[str, Dict[int, str]] = {}
    for chain_id, segment in zip(chain_ids, segments):
        full_numbering, unresolved = numbering[chain_id]
        if len(segment) != len(full_numbering):
            raise ValueError(
                f"Prompt covers {len(segment)} residue(s) of chain {chain_id} but that "
                f"chain has {len(full_numbering)}: "
                f"{len(full_numbering) - len(unresolved)} with coordinates and "
                f"{len(unresolved)} without. A prompt spans the whole chain, "
                "unresolved residues included."
            )

        chain_pins: Dict[int, str] = {}
        position = 0
        for residue, number in zip(segment, full_numbering):
            if number in unresolved:
                continue
            position += 1
            if residue != "_":
                chain_pins[position] = residue
        if chain_pins:
            pins[chain_id] = chain_pins

    return pins


def restore_unresolved_residues(
    chain_sequences: Dict[str, str],
    numbering: Dict[str, Tuple[List[int], Dict[int, str]]],
) -> str:
    """
    Threads a design's residues back into the full sequence.

    Caliby designs the resolved residues only, so a design comes back shorter
    than the sequence it was prompted with. The residues the structure does not
    resolve go back at their own positions, keeping the identity the structure
    gives them, which leaves the design the same length as every other sequence
    in a run.

    Args:
        chain_sequences: Designed sequence per chain, resolved residues only.
        numbering: Per-chain numbering, from :func:`chain_numbering`.

    Returns:
        The full designed sequence, chains joined by ``/`` in structure order.

    Raises:
        ValueError: If a designed chain does not cover that chain's resolved
            residues.
    """
    chains = []
    for chain_id in numbering:
        full_numbering, unresolved = numbering[chain_id]
        designed = chain_sequences[chain_id]
        if len(designed) + len(unresolved) != len(full_numbering):
            raise ValueError(
                f"Designed chain {chain_id} has {len(designed)} residue(s), but that "
                f"chain has {len(full_numbering) - len(unresolved)} with coordinates"
            )

        residues = []
        position = 0
        for number in full_numbering:
            if number in unresolved:
                residues.append(unresolved[number])
            else:
                residues.append(designed[position])
                position += 1
        chains.append("".join(residues))

    return "/".join(chains)


@GENERATOR_REGISTRY.register("caliby")
class CalibyGenerator(Generator):
    """Designs sequences with Caliby, through proto-tools' Caliby toolkit.

    Caliby computes Potts parameters from a backbone and samples sequences under
    the resulting energy. It can condition on a conformer ensemble instead of a
    single structure, which samples against every backbone at once. The model
    runs in the toolkit's own isolated environment, so nothing here imports
    Caliby or places it on a device.

    Which weight set to use is part of the generator's identity, not a
    parameter: each registry entry sets its own ``_CHECKPOINT``.
    """

    #: Caliby checkpoint this generator samples and scores with.
    _CHECKPOINT: str = "caliby"

    def __init__(self, seed: int = 42, device: Optional[str] = None, **kwargs: Any):
        """
        Args:
            seed: Seed forwarded to the tool.
            device: Device string forwarded to the tool. ``None`` resolves to
                whatever CUDA can see.
        """
        self.seed = seed
        self.device = device
        self.seed_rng = np.random.default_rng(seed)

    def get_extra_kwargs(self, base_variant):
        """
        Caliby requires a protein structure for generation, and conditions on an
        ensemble when the base variant carries one.
        """
        if base_variant.ensemble_structures:
            return {
                "structure_or_structures": base_variant.ensemble_structures,
                "is_ensemble": True,
            }
        return {"structure_or_structures": base_variant.structure}

    def _ensemble(
        self,
        structures: List[ProteinStructure],
    ) -> Tuple[List[Any], Dict[str, Tuple[List[int], Dict[int, str]]]]:
        """Convert an ensemble for proto-tools, and read its numbering.

        The ensemble's first member is Caliby's primary conformer: residue
        identities and numbering are read from it, and the rest are conditioned
        on.
        """
        from proto_tools.entities.structures import Structure

        # Caliby designs residues that have an alpha carbon and drops the rest,
        # so they are dropped here instead: that keeps the design the same
        # length as the structure it is mapped back onto.
        converted = [
            Structure(structure=structure.resolved_structure_pdb, structure_format="pdb")
            for structure in structures
        ]
        return converted, chain_numbering(structures[0], converted[0])

    def _config(self, config_class: Any, device: Optional[str], **kwargs: Any) -> Any:
        """Build a tool config for this generator's checkpoint and device."""
        return config_class(
            model_checkpoint=self._CHECKPOINT,
            device=resolve_device(device or self.device),
            **kwargs,
        )

    def score(
        self,
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        sequences: Union[List[str], pd.DataFrame, str] = None,
        is_ensemble: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculates the Potts energy of the sequence each structure carries.

        Caliby scores a structure's own sequence. Every supplied sequence must
        match that sequence; a mismatch raises rather than returning another
        sequence's energy.

        Args:
            structure_or_structures: One structure, one conformer ensemble with
                ``is_ensemble``, or one structure per sequence.
            sequences: Sequences to verify against the structures, or ``None``
                for one row holding each structure's own energy.
            is_ensemble: Read ``structure_or_structures`` as a single ensemble,
                the primary conformer first.
            device: Device for this call; defaults to the generator's.
        """
        from proto_tools.tools.inverse_folding.caliby import (
            CalibyScoreConfig,
            CalibyScoreInput,
            CalibyScoreStructureInput,
            run_caliby_score,
        )

        if isinstance(sequences, pd.DataFrame):
            if "sequence" not in sequences.columns:
                raise ValueError("Input dataframe must have 'sequence' column")
            seqs = sequences["sequence"].tolist()
        elif isinstance(sequences, str):
            seqs = [sequences]
        else:
            seqs = sequences

        # Every case is a list of ensembles to score, so that one structure, an
        # ensemble and one structure per sequence share a single dispatch.
        if is_ensemble:
            ensembles = [as_ensemble(structure_or_structures)]
        elif is_single_structure(structure_or_structures):
            ensembles = [[ProteinStructure(structure_or_structures)]]
        else:
            ensembles = [[ProteinStructure(s)] for s in structure_or_structures]
            if seqs is None or len(ensembles) != len(seqs):
                raise ValueError(
                    "Number of structures must match sequences if list of structures provided."
                )

        inputs, numberings = [], []
        for ensemble in ensembles:
            converted, numbering = self._ensemble(ensemble)
            inputs.append(
                CalibyScoreStructureInput(
                    structure=converted[0], conformers=converted[1:]
                )
            )
            numberings.append(numbering)

        scores = run_caliby_score(
            CalibyScoreInput(inputs=inputs),
            self._config(CalibyScoreConfig, device),
        ).scores

        energies = [float(score["potts_energy"]) for score in scores]
        scored = [
            # Caliby reports the scored sequence with ':' between chains, in
            # chain ID order, and covering resolved residues only.
            restore_unresolved_residues(
                dict(zip(sorted(numbering), score.sequence.split(":"))), numbering
            )
            for score, numbering in zip(scores, numberings)
        ]

        if seqs is not None:
            if len(energies) == 1 and len(seqs) > 1:
                energies = energies * len(seqs)
                scored = scored * len(seqs)
            if len(seqs) != len(energies):
                raise ValueError("Caliby must return one score per requested sequence")
            if any(
                requested != carried for requested, carried in zip(seqs, scored)
            ):
                raise ValueError(
                    "Caliby can only score the sequence its structure carries; "
                    "the requested sequence differs. Supply a matching structure."
                )

        return pd.DataFrame({"caliby_energy_score": energies})

    def generate(
        self,
        prompts: Union[List[str], pd.DataFrame],
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        generations_per_sequence: int = 10,
        is_ensemble: bool = False,
        **kwargs,
    ):
        """
        Caliby can produce multiple designs per prompt, so each prompt's row is
        repeated once per design rather than once per prompt.

        Args:
            prompts: Prompts, or a dataframe with a ``prompt`` column.
            structure_or_structures: One structure for every prompt, one
                structure per prompt, or - with ``is_ensemble`` - one conformer
                ensemble used for every prompt.
            generations_per_sequence: Designs to sample per prompt.
            is_ensemble: Read ``structure_or_structures`` as a single ensemble,
                the primary conformer first, rather than one structure per
                prompt.
        """
        if isinstance(prompts, list):
            prompts = pd.DataFrame({"prompt": prompts})

        if "prompt" not in prompts.columns:
            raise ValueError("Input dataframe to generator must contain a 'prompt' column")

        if is_ensemble:
            ensembles = [as_ensemble(structure_or_structures)] * len(prompts)
        else:
            ensembles = [
                [structure]
                for structure in as_structures(structure_or_structures, len(prompts))
            ]

        all_generations = []
        for position, (_index, row) in enumerate(
            tqdm(
                prompts.iterrows(),
                desc="Generating sequences with Caliby",
                total=len(prompts),
                unit="prompts",
            )
        ):
            generations = self._generate(
                prompts=[row["prompt"]],
                structure_or_structures=ensembles[position],
                generations_per_sequence=generations_per_sequence,
                **kwargs,
            )
            all_generations.extend(rows_with_generations(row, generations))

        return pd.DataFrame(all_generations)

    def _generate(
        self,
        prompts: List[str],
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        generations_per_sequence: int = 10,
        temperature: float = 0.01,
        excluded_amino_acids: Optional[List[str]] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Design sequences for each prompt.

        Args:
            prompts: Sequences with ``_`` at the positions to design, chains
                split by ``/``. Positions that are not masked are pinned to the
                residue the prompt carries.
            structure_or_structures: The backbone to design on. A list is a
                conformer ensemble to condition on, the primary conformer first,
                and is used for every prompt - unlike
                :meth:`generate`, which also accepts one structure per prompt.
            generations_per_sequence: Designs to sample per prompt.
            temperature: Final Potts sampling temperature, annealed from 1.0.
            excluded_amino_acids: Single-letter codes the model may not place at
                designed positions, e.g. ``["C"]``.
            device: Device for this call; defaults to the generator's.

        Returns:
            One dict per design with ``sequence``, ``prompt`` and
            ``caliby_energy_score`` keys.
        """
        from proto_tools.tools.inverse_folding.caliby import (
            CalibySampleConfig,
            CalibySampleInput,
            CalibyStructureInput,
            run_caliby_sample,
        )

        converted, numbering = self._ensemble(as_ensemble(structure_or_structures))
        config = self._config(
            CalibySampleConfig,
            device,
            num_sequences_per_structure=generations_per_sequence,
            temperature=temperature,
            excluded_amino_acids=excluded_amino_acids,
            seed=int(self.seed_rng.integers(0, 2**31 - 1)),
        )

        output = run_caliby_sample(
            CalibySampleInput(
                inputs=[
                    CalibyStructureInput(
                        structure=converted[0],
                        conformers=converted[1:],
                        pinned_residues=prompt_pins(prompt, numbering) or None,
                    )
                    for prompt in prompts
                ]
            ),
            config,
        )

        generations = []
        for prompt, design_set in zip(prompts, output.design_sets, strict=True):
            for design in design_set.complexes:
                chain_sequences = {chain.id: chain.sequence for chain in design.chains}
                sequence = restore_unresolved_residues(chain_sequences, numbering)
                if len(sequence) != len(prompt) or any(
                    fixed != "_" and fixed != residue for fixed, residue in zip(prompt, sequence)
                ):
                    raise ValueError("Caliby output changed an unmasked residue or chain layout")
                generations.append(
                    {
                        "sequence": sequence,
                        "prompt": prompt,
                        "caliby_energy_score": float(design.metrics["potts_energy"]),
                    }
                )

        return generations


@GENERATOR_REGISTRY.register("soluble_caliby")
class SolubleCalibyGenerator(CalibyGenerator):
    """Caliby with the soluble-protein weight set.

    ``soluble_caliby_v1`` is the current soluble checkpoint; the older
    ``soluble_caliby`` is a different model and is not what this generator
    loads.
    """

    _CHECKPOINT = "soluble_caliby_v1"
