"""
base_variant.py

Contains the BaseVariant class, which is used to contain information about the starting variant for the sampler.
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from provada.sequences.vocab import amino_acid_sequence_check
from provada.sequences.io import get_fixed_positions_from_file, get_sequence
from provada.utils.log import get_logger
from omegaconf import DictConfig
import yaml
from provada.utils.structure import ProteinStructure

logger = get_logger(__name__)


@dataclass
class BaseVariant:
    """
    Class to contain information about the starting variant for the sampler.

    Args:
        sequence: The starting sequence
        structure: Optional structure to use for structure-based generation
        fixed_indices: The indices of the fixed positions in the sequence (0-indexed)

        score_dict: A dictionary of scores for the starting variant
    """

    sequence: Optional[str] = None
    structure: Optional[ProteinStructure] = None
    fixed_indices: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.sequence is not None:
            self._validate_sequence(self.sequence)
        if self.structure is not None:
            self.structure = ProteinStructure(self.structure)

        self.score_dict = {}
        self.normalized_score_dict = {}

    def _validate_sequence(self, sequence: str):
        """Validate amino acid sequence."""
        if not amino_acid_sequence_check(sequence, allow_unknown=True, allow_stop=False):
            raise ValueError(f"Sequence is not a valid amino acid sequence: {sequence}")
        logger.debug(f"Set sequence to: {sequence}")

    def set_sequence(self, sequence: str):
        """Set and validate sequence."""
        self._validate_sequence(sequence)
        self.sequence = sequence

    def set_structure(self, structure: Union[str, Path, ProteinStructure]):
        """Set and validate PDB path."""
        self.structure = ProteinStructure(structure)

    def set_fixed_indices(self, fixed_indices: List[int]):
        """Set fixed indices."""
        self.fixed_indices = fixed_indices
        logger.debug(f"Set {len(fixed_indices)} fixed indices")

    def load_sequence(self, sequence_file_path: str):
        """Load the sequence from a file."""
        sequence = get_sequence(sequence_file_path)
        self.set_sequence(sequence)

    def load_structure(self, structure_content_or_path: Union[str, Path, ProteinStructure]):
        """Load the structure."""
        self.structure = ProteinStructure(structure_content_or_path)

    def load_fixed_position_files(
        self, fixed_position_files: List[str], assume_1_indexed: bool = True
    ):
        """Load the fixed positions from files."""
        fixed_positions = []
        for fixed_position_file in fixed_position_files:
            fixed_positions.extend(
                get_fixed_positions_from_file(fixed_position_file, assume_1_indexed)
            )
        self.set_fixed_indices(sorted(fixed_positions))

    def load_base_variant_from_config(self, config: DictConfig):
        """Load the base variant from a configuration dictionary."""
        if "sequence_file_path" in config:
            self.load_sequence(config.sequence_file_path)
        if "structure_file_path" in config:
            self.load_structure(config.structure_file_path)
        if "fixed_position_files" in config:
            self.load_fixed_position_files(config.fixed_position_files)

    def to_dict(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Convert the base variant to a dictionary."""

        if prefix is None:
            prefix = ""

        # Ensure prefix ends with exactly one dot
        prefix = prefix.rstrip(".") + "." if prefix else ""

        output_dictionary = {
            f"{prefix}sequence": self.sequence,
            f"{prefix}structure": self.structure,
            f"{prefix}fixed_indices": self.fixed_indices,
        }

        if len(self.score_dict) > 0:
            for score_key, score_value in self.score_dict.items():
                output_dictionary[f"{prefix}score." + score_key] = score_value
        if len(self.normalized_score_dict) > 0:
            for score_key, score_value in self.normalized_score_dict.items():
                output_dictionary[f"{prefix}normalized_score." + score_key] = score_value

        return output_dictionary

    def save_yaml(self, path: str):
        """Save the base variant to a yaml file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @property
    def sequence_length(self) -> int:
        """Get the length of the sequence."""
        return len(self.sequence)

    @property
    def fixed_position_indices(self) -> List[int]:
        """Get the fixed position indices."""
        return self.fixed_indices

    @property
    def num_designable_positions(self) -> int:
        """Get the number of designable positions."""
        return self.sequence_length - len(self.fixed_position_indices)
