"""
base_variant.py

Contains the BaseVariant class, which is used to contain information about the starting variant for the sampler.
"""

from typing import TYPE_CHECKING, List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from provada.sequences.vocab import amino_acid_sequence_check
from provada.sequences.io import get_fixed_positions_from_file, get_sequence
from provada.utils.log import get_logger
import yaml
from provada.utils.structure import ProteinStructure
import gzip
from numbers import Integral

if TYPE_CHECKING:
    from provada.config import BaseVariantConfig

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
    ensemble_structures: List[ProteinStructure] = field(default_factory=list)
    fixed_indices: Union[List[int], Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.sequence is not None:
            self._validate_sequence(self.sequence)
        if self.structure is not None:
            self.structure = ProteinStructure(self.structure)
        if self.ensemble_structures:
            self.ensemble_structures = [ProteinStructure(s) for s in self.ensemble_structures]

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
        
        # Check if multichain sequence matches structure chains if structure is set
        if self.structure:
            structure_chain_count = len(self.structure.get_chain_sequences())
            sequence_chain_count = sequence.count('/') + 1
            if sequence_chain_count != structure_chain_count:
                # Fallback: maybe sequence is single chain but structure has 1 chain?
                # If sequence has no '/', count is 1.
                if structure_chain_count > 1 and sequence_chain_count == 1:
                     raise ValueError(f"Structure has {structure_chain_count} chains but sequence appears to be single chain (no '/'). Please provide multi-chain sequence separated by '/'.")
                elif structure_chain_count != sequence_chain_count:
                     raise ValueError(f"Sequence has {sequence_chain_count} chains but structure has {structure_chain_count} chains.")

        self.sequence = sequence

    def set_structure(self, structure: Union[str, Path, ProteinStructure]):
        """Set and validate PDB path."""
        self.structure = ProteinStructure(structure)
        if self.sequence is not None:
            self.set_sequence(self.sequence)

    def load_ensemble_structures(self, ensemble_dir: Union[str, Path], main_conformer: str):
        """Load the ensemble structures from a directory."""
        from provada.utils.structure import SUPPORTED_EXTENSIONS
        ensemble_dir = Path(str(ensemble_dir))
        if not ensemble_dir.is_dir():
            raise ValueError(f"Ensemble directory {ensemble_dir} is not a directory.")
        
        main_conformer = str(main_conformer)
        main_path = ensemble_dir / main_conformer
        if not main_path.exists():
            raise ValueError(f"Main conformer {main_conformer} not found in {ensemble_dir}.")
        
        self.set_structure(main_path)
        
        # Load all other structures
        self.ensemble_structures = [self.structure]
        for struct_file in ensemble_dir.iterdir():
            if (
                struct_file.is_file() 
                and struct_file.name != main_conformer 
                and struct_file.suffix[1:].lower() in SUPPORTED_EXTENSIONS
            ):
                self.ensemble_structures.append(ProteinStructure(struct_file))
        
        logger.info(f"Loaded {len(self.ensemble_structures)} ensemble structures from {ensemble_dir}")

    def get_flattened_fixed_indices(self) -> List[int]:
        """Map zero-based constraints in structure chain order, fixing separators."""
        if self.sequence is None:
            if isinstance(self.fixed_indices, list) and not self.fixed_indices:
                return []
            raise ValueError("Cannot validate fixed indices without a sequence")
        flat = {i for i, residue in enumerate(self.sequence) if residue == "/"}
        if isinstance(self.fixed_indices, list):
            for position in self.fixed_indices:
                if not isinstance(position, Integral) or isinstance(position, bool) or not 0 <= position < len(self.sequence):
                    raise ValueError(f"Fixed index {position!r} is outside the sequence")
                flat.add(int(position))
        elif isinstance(self.fixed_indices, dict):
            if self.structure is None:
                raise ValueError("Chain-based fixed indices require a structure to define chain order")
            chain_ids = self.structure.get_chain_ids()
            segments = self.sequence.split("/")
            if len(chain_ids) != len(segments):
                raise ValueError("Sequence and structure must have the same chains")
            unknown = set(self.fixed_indices) - set(chain_ids)
            if unknown:
                raise ValueError(f"Fixed position chains not found in structure: {sorted(unknown)}")
            offset = 0
            for chain_id, segment in zip(chain_ids, segments):
                positions = self.fixed_indices.get(chain_id, [])
                if positions == "ALL":
                    positions = range(len(segment))
                for position in positions:
                    if not isinstance(position, Integral) or isinstance(position, bool) or not 0 <= position < len(segment):
                        raise ValueError(f"Fixed index {position!r} is outside chain {chain_id}")
                    flat.add(offset + int(position))
                offset += len(segment) + 1
        else:
            raise ValueError("Fixed indices must be a list or a chain-to-positions mapping")
        return sorted(flat)

    def set_fixed_indices(self, fixed_indices: Union[List[int], Dict[str, Any]]):
        """Set fixed indices."""
        self.fixed_indices = fixed_indices
        if isinstance(fixed_indices, list):
            logger.debug(f"Set {len(fixed_indices)} fixed indices")
        else:
            logger.debug(f"Set fixed indices for chains: {list(fixed_indices.keys())}")

    def load_sequence(self, sequence_file_path: str):
        """Load the sequence from a file."""
        chain_ids = self.structure.get_chain_ids() if self.structure is not None else None
        sequence = get_sequence(sequence_file_path, chain_ids=chain_ids)
        self.set_sequence(sequence)

    def load_structure(self, structure_content_or_path: Union[str, Path, ProteinStructure]):
        """Load the structure."""
        self.set_structure(structure_content_or_path)

    def load_fixed_position_files(
        self, fixed_position_files: List[str], assume_1_indexed: bool = True
    ):
        """Load the fixed positions from files."""
        accumulated_list = []
        accumulated_dict = {}
        has_list = False
        has_dict = False

        for fixed_position_file in fixed_position_files:
            res = get_fixed_positions_from_file(fixed_position_file, assume_1_indexed)
            if isinstance(res, list):
                accumulated_list.extend(res)
                has_list = True
            elif isinstance(res, dict):
                 # Merge dicts
                 for chain, val in res.items():
                     if val == "ALL":
                         accumulated_dict[chain] = "ALL"
                     elif chain not in accumulated_dict:
                         accumulated_dict[chain] = val
                     elif accumulated_dict[chain] != "ALL":
                         accumulated_dict[chain].extend(val)
                 has_dict = True
        
        if has_list and has_dict:
             raise ValueError("Cannot mix single-chain (integer list) and multi-chain (chain-specified) fixed position files.")
        
        if has_dict:
            self.set_fixed_indices({chain: value if value == "ALL" else sorted(set(value))
                                    for chain, value in accumulated_dict.items()})
        else:
            self.set_fixed_indices(sorted(set(accumulated_list)))
        self.get_flattened_fixed_indices()

    def load_mhc_alleles(self, mhc_alleles: List[str]):
        """
        Given a list of allele names, store the formatted allele names.
        """
        from proto_tools.tools import normalize_alleles

        mhc_alleles = mhc_alleles.split(",") if isinstance(mhc_alleles, str) else mhc_alleles
        # proto-tools normalizes to the spelling netMHCpan accepts and validates
        # against the install's own allele list, e.g. "HLA-A02:01".
        self.mhc_alleles = normalize_alleles(mhc_alleles)

    def load_self_peptide_db(self, self_peptide_db_path: Optional[str] = None):
        """
        Load the peptides to treat as self from a file (plain text or gzip),
        one peptide per line.

        Args:
            self_peptide_db_path: File to read, or ``None`` to use the managed
                human 9mer peptidome, downloading it on first use.

        Raises:
            ValueError: If the database holds no peptides of the length ProVADA
                scans, which would leave every epitope counted as foreign.
        """
        from provada.components.evaluator import PEPTIDE_LENGTH
        from provada.utils.datasets import HUMAN_PEPTIDOME_9MER, resolve_dataset

        if self_peptide_db_path is None:
            self_peptide_db_path = str(resolve_dataset(HUMAN_PEPTIDOME_9MER))

        opener = gzip.open if self_peptide_db_path.endswith(('.gz', '.gzip')) else open
        self_peptide_db = set()
        with opener(self_peptide_db_path, 'rt') as f:
            for line in f:
                peptide = line.strip()
                if peptide:
                    self_peptide_db.add(peptide)

        lengths = {len(peptide) for peptide in self_peptide_db}
        if PEPTIDE_LENGTH not in lengths:
            raise ValueError(
                f"{self_peptide_db_path} holds peptides of length(s) {sorted(lengths)}, "
                f"none of them {PEPTIDE_LENGTH}, which is the length ProVADA scans. "
                "Every predicted epitope would count as foreign, so immunogenicity "
                "would be scored as though nothing were excluded."
            )

        logger.info(
            f"Loaded {len(self_peptide_db)} self peptides of length(s) "
            f"{sorted(lengths)} from {self_peptide_db_path}"
        )
        self.self_peptide_db = self_peptide_db

    def load_base_variant_from_config(self, config: "BaseVariantConfig"):
        """Load the base variant from its config.

        Args:
            config (BaseVariantConfig): Paths describing the starting protein.
                ``sequence_file_path`` is required by the schema; everything
                else is optional. An ensemble takes precedence over a single
                structure, matching the previous dict-based behavior.
        """
        if config.ensemble_structure_dir is not None and config.main_conformer is not None:
            self.load_ensemble_structures(
                config.ensemble_structure_dir, config.main_conformer
            )
        elif config.structure_file_path is not None:
            self.load_structure(config.structure_file_path)

        self.load_sequence(config.sequence_file_path)

        if config.fixed_position_files:
            self.load_fixed_position_files(config.fixed_position_files)
        if config.mhc_alleles is not None:
            self.load_mhc_alleles(config.mhc_alleles)
        if config.exclude_self_peptides:
            self.load_self_peptide_db(config.self_peptide_db_path)

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
    def designable_chain_index(self) -> int:
        """
        Identify the single designable chain index in the sequence. 
        Raises an error if multiple or zero designable chains are found.
        Returns the 0-based index of the chain among strings split by '/'.
        """
        if not self.sequence:
            raise ValueError("Sequence not set")
            
        chains = self.sequence.split('/')
        if len(chains) == 1:
            return 0
            
        # Determine which chain is designable based on fixed_indices
        # If fixed_indices is a dict, look for keys that are NOT set to "ALL"
        # Since we map dict keys typically by chain ID, mapping them to index requires structure or assumption.
        
        # Strategy: Use flattened_fixed_indices.
        # Iterate over chains and check if all indices of that chain are in flattened_fixed_indices.
        
        flattened = set(self.get_flattened_fixed_indices())
        designable_indices = []
        
        current_offset = 0
        for i, chain_seq in enumerate(chains):
            # Calculate range for this chain in the FULL sequence (including separator offsets)
            # Full String: "SEQ1/SEQ2"
            # Idx calculation:
            chain_start = current_offset
            chain_end = current_offset + len(chain_seq)
            
            # Check if all indices in [chain_start, chain_end) are in flattened
            chain_indices = set(range(chain_start, chain_end))
            
            # If the intersection is smaller than the chain_indices, it means some residues are NOT fixed.
            # Thus this chain is designable.
            if not chain_indices.issubset(flattened):
                designable_indices.append(i)
                
            # Update offset: length of chain + 1 for '/'
            current_offset += len(chain_seq) + 1
            
        if len(designable_indices) != 1:
            raise ValueError(f"Expected exactly 1 designable chain, found {len(designable_indices)}: {designable_indices}. (Indices: {designable_indices})")
            
        return designable_indices[0]

    @property
    def fixed_position_indices(self) -> List[int]:
        """Get the flattened fixed position indices."""
        return self.get_flattened_fixed_indices()

    @property
    def num_designable_positions(self) -> int:
        """Get the number of designable positions."""
        return self.sequence_length - len(self.fixed_position_indices)
