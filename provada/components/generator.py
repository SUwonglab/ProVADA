"""
generator.py
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any
from pathlib import Path
import torch
from provada.models.mpnn import ProteinMPNNModel
from provada.models.esm3 import ESM3Model, ESMProtein, GenerationConfig
from provada.utils.env import device_manager, suppress_console_output
from provada.sequences.mutate import sample_amino_acid
from provada.utils.setup import seed_everything
import pandas as pd
from provada.utils.registry import GENERATOR_REGISTRY
from tqdm import tqdm
from provada.utils.structure import ProteinStructure
from provada.utils.log import get_logger

logger = get_logger(__name__)


def get_generator(name: str, **kwargs):
    """
    Get a generator instance from the registry.
    """
    return GENERATOR_REGISTRY.get_instance(name, **kwargs)


class Generator(ABC):
    """
    Generator is an abstract class for all generators.
    """

    def generate(self, prompts: Union[List[str], pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Assumes that the input sequences are masked sequences and that the output
        sequences are unmasked sequences.
        """
        if isinstance(prompts, list):
            prompts = pd.DataFrame({"prompt": prompts})
        else:
            prompts = prompts.copy()
            # Ensure that the "prompt" column exists
            if "prompt" not in prompts.columns:
                raise ValueError("Input dataframe to generator must contain a 'prompt' column")

        # Call the generator
        try:
            generation_output = pd.DataFrame(
                self._generate(prompts["prompt"].tolist(), **kwargs)
            )
        except Exception as e:
            logger.error(f"Error in _generate function of {self.__class__.__name__}: {e}")
            raise e

        # If the input was a dataframe, return with the original columns included
        if isinstance(prompts, pd.DataFrame):
            output = pd.concat([prompts, generation_output], axis=1)
            # Drop duplicate columns
            output = output.loc[:, ~output.columns.duplicated()]
            return output
        else:
            return generation_output

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


@GENERATOR_REGISTRY.register("esm3")
class ESM3Generator(Generator):
    """
    ESM3Generator is a generator that uses the ESM3 model to generate sequences.
    """

    def __init__(self, seed: int = 42):
        """
        Set up the ESM3 model.
        """

        # Load the ESM3 model
        self.esm3 = ESM3Model()
        seed_everything(seed)

    def _generate(
        self,
        prompts: List[str],
        max_batch_size: int = 20,
        device: str = device_manager.get_device(),
        schedule: str = "cosine",
        strategy: str = "entropy",
        num_steps: int = 16,
        temperature: float = 0.8,
        temperature_annealing: bool = True,
        top_p: float = 0.95,
        omit_AAs: Optional[str] = None,
    ):
        """
        Generate sequences using the ESM3 model. For now, this utilizes the
        built-in ESM3 generation API.

        Args:
            prompts (List[str]): The sequences to generate from
            max_batch_size (int): The maximum batch size to use for generation
            device (str): The device to use for generation
            schedule (str): Controls the unmasking schedule during iterative
                generation. Options "cosine" and "linear"
            strategy (str): Controls which tokens to unmask at each step. Options
                "random" and "entropy"
            num_steps: The number of iterative decoding steps. This is automatically
                capped at the minimum number of masked positions in a sequence by
                the ESM3 generation API.
            temperature: The temperature to use for generation.
            temperature_annealing: Whether to anneal the temperature over the course
                of generation.
            top_p: Nucleus sampling threshold. Lower values means more conservative
                sampling.
            omit_AAs: A string of amino acids to omit from the design.
                (e.g. "CP" to omit cysteine and proline)

        Returns:
            List[str]: The generated sequences
        """

        # Create the list of invalid ids
        if omit_AAs is None:
            omit_AAs = ""

        # Always omit X
        if "X" not in omit_AAs:
            omit_AAs = omit_AAs + "X"

        # Create the list of invalid ids
        invalid_ids = [self.esm3.tokenizer.get_vocab()[aa] for aa in list(omit_AAs)]

        # Create the generation config from the provided parameters
        generation_config = GenerationConfig(
            track="sequence",  # Specifies which protein track to generate
            schedule=schedule,
            strategy=strategy,
            num_steps=num_steps,
            temperature=temperature,
            temperature_annealing=temperature_annealing,
            top_p=top_p,
            invalid_ids=invalid_ids,
        )

        # Move the model to the active device
        self.esm3.model.to(device)

        # Create batches of sequences based on the max batch size
        max_batch_size = min(max_batch_size, len(prompts))
        batches = [
            prompts[i : i + max_batch_size] for i in range(0, len(prompts), max_batch_size)
        ]

        # List to store the generations
        generations = []

        # Generate sequences for each batch
        for batch in tqdm(
            batches,
            desc="Generating sequences with ESM3",
            unit="batch",
            total=len(batches),
        ):
            # Create ESM3Protein objects for each sequence
            esm3_proteins = [ESMProtein(sequence=seq) for seq in batch]
            gen_configs = [generation_config] * len(esm3_proteins)

            with suppress_console_output(stdout=True, stderr=True):
                results = self.esm3.model.batch_generate(
                    inputs=esm3_proteins,
                    configs=gen_configs,
                )

            # Add the generations to the list of all generations
            generations.extend([res.sequence for res in results])

        # Move the model off device
        self.esm3.model.to("cpu")

        return [
            {"sequence": gen, "prompt": prompt} for gen, prompt in zip(generations, prompts)
        ]


@GENERATOR_REGISTRY.register("mpnn")
class MPNNGenerator(Generator):
    """
    MPNNGenerator
    """

    def __init__(self, seed: int = 42):
        """
        Set up the MPNN generator
        """

        self.seed_rng = np.random.default_rng(seed)
        self.mpnn_model = ProteinMPNNModel(seed=seed, use_soluble_model=False)

    def get_extra_kwargs(self, base_variant):
        """
        MPNNGenerator requires a protein structure for generation.
        """
        return {"structure_or_structures": base_variant.structure}

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
        ProteinMPNN can produce multiple generations per sequence. Here we need
        a modified version of the generate method that can handle this for
        dataframe inputs to ensure that included fields are consistent with the
        inputs.
        """

        # Convert list to DataFrame if needed
        if isinstance(prompts, list):
            prompts = pd.DataFrame({"prompt": prompts})

        # If a single pdb path is provided, use it for generations for all sequences
        if (
            type(structure_or_structures) == str
            or isinstance(structure_or_structures, Path)
            or isinstance(structure_or_structures, ProteinStructure)
        ):
            structures = [ProteinStructure(structure_or_structures)] * len(prompts)
        else:
            structures = [ProteinStructure(structure) for structure in structure_or_structures]

        # Ensure that the "prompt" column exists
        if "prompt" not in prompts.columns:
            raise ValueError("Input dataframe to generator must contain a 'prompt' column")

        sequence_df = prompts.copy()
        all_generations = []
        for idx, row in tqdm(
            sequence_df.iterrows(),
            desc="Generating sequences with MPNN",
            total=len(sequence_df),
            unit="prompts",
        ):
            # Extract the sequence and pdb path
            prompt = row["prompt"]
            structure = structures[idx]

            # Generate the generations
            generations = self._generate(
                prompts=[prompt],
                structure_or_structures=structure,
                generations_per_sequence=generations_per_sequence,
                **kwargs,
            )

            # For each generation, create a new row with all original metadata
            # Indexing issue fixed by converting to object type
            for generation in generations:
                # Make a copy of the row
                new_row = (
                    row.copy()
                )  # <- This itself also copies the index of the row, causing problems
                new_row = row.astype(
                    object
                )  # converts all fields to object type, preventing issues when adding new fields

                # Create a new row with the generation data
                for key, value in generation.items():
                    new_row[key] = value

                new_row.name = None  # prevents carrying over the index
                # Add the new row to the list of all generations
                all_generations.append(new_row)

        # Return the dataframe of all generations
        return pd.DataFrame(all_generations)

    def _generate(
        self,
        prompts: List[str],
        structure_or_structures: Union[
            str, Path, ProteinStructure, List[Union[str, Path, ProteinStructure]]
        ],
        generations_per_sequence: int = 10,
        temperature: float = 0.8,
        omit_AAs: Optional[str] = None,
        batch_size: int = 50,
        **kwargs: Any,
    ) -> list[dict]:

        # If a single pdb path is provided, use it for all sequences
        if (
            type(structure_or_structures) == str
            or isinstance(structure_or_structures, Path)
            or isinstance(structure_or_structures, ProteinStructure)
        ):
            structures = [ProteinStructure(structure_or_structures)] * len(prompts)
        else:
            structures = [ProteinStructure(structure) for structure in structure_or_structures]

        # Generate sequences for each sequence
        generations = []

        keep_on_device_flags = [True] * len(prompts)
        keep_on_device_flags[-1] = False

        for prompt, structure, keep_on_device in zip(
            prompts, structures, keep_on_device_flags
        ):
            new_generations = []

            fixed_positions = []
            for ind, aa in enumerate(prompt):
                if aa != "_":
                    fixed_positions.append(ind + 1)

            incoming = self.mpnn_model.sample(
                structure=structure,
                num_sequences_to_generate=generations_per_sequence,
                batch_size=batch_size,
                temperature=temperature,
                fixed_positions=fixed_positions,
                omit_AAs=omit_AAs,
                keep_on_gpu=keep_on_device,
                prompt=prompt,
            )

            for gen_ind in range(generations_per_sequence):
                new_generations.append(
                    {
                        "sequence": incoming["filled_seqs"][gen_ind],
                        "prompt": prompt,
                    }
                )
                new_generations[gen_ind]["mpnn_score"] = incoming["scores"][gen_ind]

            generations.extend(new_generations)

        # Ensure GPU memory is freed after all generations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generations


@GENERATOR_REGISTRY.register("soluble_mpnn")
class SolubleMPNNGenerator(MPNNGenerator):

    def __init__(self, seed: int = 42):
        """
        Set up the soluble MPNN generator
        """
        self.seed_rng = np.random.default_rng(seed)
        self.mpnn_model = ProteinMPNNModel(seed=seed, use_soluble_model=True)


@GENERATOR_REGISTRY.register("random")
class RandomGenerator(Generator):

    def __init__(self, seed: int = 42):
        """
        Set up the random generator
        """

        self.seed_rng = np.random.default_rng(seed)

    def _generate(self, prompts: List[str], codon_scheme: str = "NNN") -> list[dict]:
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
