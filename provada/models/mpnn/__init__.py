"""
mpnn/__init__.py

ProteinMPNN implementation
"""

from typing import List, Optional, Dict, Union
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from provada.utils.log import get_logger
from provada.utils.structure import ProteinStructure
from provada.utils.env import device_manager
from protein_mpnn import ProteinMPNN

logger = get_logger(__name__)


class ProteinMPNNModel:
    def __init__(
        self,
        seed: int = 42,
        model_name: str = "v_48_020",
        ca_only: bool = False,
        use_soluble_model: bool = False,
    ):
        logger.info("Loading ProteinMPNN model")

        self.model = ProteinMPNN(
            model_name=model_name,
            ca_only=ca_only,
            use_soluble_model=use_soluble_model,
            device="cpu",
        )
        self.device = "cpu"

        self.seed = seed
        self.seed_rng = np.random.default_rng(seed)
        self.alphabet = "ACDEFGHIKLMNPQRSTVWYX"

    def sample(
        self,
        structure: Union[str, Path, ProteinStructure],
        num_sequences_to_generate: int = 10,
        batch_size: int = 10,
        temperature: float = 0.2,
        fixed_positions: Optional[Union[Dict[str, List[int]], List[int]]] = None,
        omit_AAs: Optional[str] = None,
        device: str = None,
        keep_on_gpu: bool = False,
        prompt: Optional[str] = None,
    ):
        """
        Samples sequences using the ProteinMPNN model

        Args:
            structure: PDB structure (path, string, or ProteinStructure object)
            num_sequences_to_generate: Number of sequences to generate
            temperature: Temperature for sampling
            fixed_positions: Fixed positions for sampling
                - If a dictionary, the keys are the chain IDs and the values are lists of residue indices (1-indexed).
                - If a list, the values are residue indices (1-indexed). Expects only one chain.
            omit_AAs: Amino acids to omit from sampling
            device: Device to use (defaults to device_manager.get_device())
            keep_on_gpu: Keep the model on GPU
            prompt: Optional masked sequence prompt with '_' for masked positions.
                If provided, fixed positions in the prompt will override the sequence
                in the PDB structure before passing to ProteinMPNN.
        """
        # Use device_manager if no device specified
        if device is None:
            device = device_manager.get_device()

        # Lazy load the model to the correct device
        self.to_device(device)

        # Read in PDB structure
        structure = ProteinStructure(structure)
        chain_dict = structure.get_chain_sequences()

        # Convert the fixed positions to a dictionary if it is a list
        if isinstance(fixed_positions, list):
            if len(chain_dict) > 1:
                raise ValueError(
                    "If multiple chains are present, fixed_positions must be a dictionary with chain IDs as keys."
                )
            fixed_positions = {list(chain_dict.keys())[0]: fixed_positions}

        # Generate seed for this sampling run
        seed = int(self.seed_rng.integers(0, np.iinfo(np.int32).max))

        # Nest fixed_positions under the predictable protein name "protein"
        # The ProteinMPNN wrapper now uses "protein" as the name for PDB string inputs
        if fixed_positions is not None:
            fixed_positions_nested = {"protein": fixed_positions}
        else:
            fixed_positions_nested = None

        if omit_AAs is None:
            omit_AAs = "X"
        else:
            if "X" not in omit_AAs:
                omit_AAs = omit_AAs + "X"

        # Since we override sequence within mpnn wrapper, no need to override at pdb level any more!
        pdb_str_to_use = structure.structure_pdb

        # convert prompt to prompt_dict where key is the chain id
        prompt_dict = None
        if prompt is not None:
            if len(chain_dict) > 1:
                raise ValueError("Sequence prompt only supports single-chain structures.")
            chain_id = list(chain_dict.keys())[0]
            prompt_dict = {chain_id: prompt}

        # Sample Sequences using underlying ProteinMPNN API
        results = self.model.sample(
            pdb_path_or_str=pdb_str_to_use,
            prompt=prompt_dict,
            num_seq_per_target=num_sequences_to_generate,
            batch_size=batch_size,
            sampling_temp=str(temperature),
            seed=seed,
            fixed_positions_dict=fixed_positions_nested,
            omit_AAs=omit_AAs,
        )

        # Extract results from nested dictionary
        # The results are keyed by protein name, get the first (and only) entry
        protein_results = list(results.values())[0]

        return_dict = {
            "filled_seqs": protein_results["sequences"],
            "scores": protein_results["scores"].tolist(),
        }

        if not keep_on_gpu:
            self.to_device("cpu")
            # Explicitly free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return return_dict

    def score(
        self,
        structure: Union[str, Path, ProteinStructure],
        sequences: List[str],
        fixed_positions: Optional[Union[Dict[str, List[int]], List[int]]] = None,
        device: str = None,
        keep_on_gpu: bool = False,
    ):
        """
        Scores sequences using the ProteinMPNN model

        Args:
            structure: PDB structure (path, string, or ProteinStructure object)
            sequences: List of sequences to score
            fixed_positions: Fixed positions for scoring
            device: Device to use (defaults to device_manager.get_device())
            keep_on_gpu: Keep the model on GPU
        """
        # Use device_manager if no device specified
        if device is None:
            device = device_manager.get_device()

        # Lazy load the model to the correct device
        self.to_device(device)

        structure = ProteinStructure(structure)
        chain_dict = structure.get_chain_sequences()

        # Convert the fixed positions to a dictionary if it is a list
        if isinstance(fixed_positions, list):
            if len(chain_dict) > 1:
                raise ValueError(
                    "If multiple chains are present, fixed_positions must be a dictionary with chain IDs as keys."
                )
            fixed_positions = {list(chain_dict.keys())[0]: fixed_positions}

        # Write sequences to a temporary fasta file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            fasta_path = f.name
            for i, sequence in enumerate(sequences):
                f.write(f">seq_{i}\n{sequence}\n")

        try:
            # Generate seed for this scoring run
            seed = int(self.seed_rng.integers(0, np.iinfo(np.int32).max))

            # Score sequences using new API
            results = self.model.score(
                pdb_path_or_str=structure.structure_pdb,
                fasta_path=fasta_path,
                seed=seed,
                fixed_positions_dict=fixed_positions,
            )

            # Extract results from nested dictionary
            protein_results = list(results.values())[0]

            # Extract scores from fasta_scores list
            scores = [
                float(fasta_score["mean_score"])
                for fasta_score in protein_results["fasta_scores"]
            ]

        finally:
            # Clean up temporary fasta file
            if os.path.exists(fasta_path):
                os.unlink(fasta_path)

        if not keep_on_gpu:
            self.to_device("cpu")
            # Explicitly free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return scores

    def to_device(self, device: str):
        """
        Moves the model to the specified device
        """
        if self.device == device:
            return

        self.model.to_device(device)
        self.device = device
