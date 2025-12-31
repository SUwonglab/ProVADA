"""
structure.py

Class for handling structures
"""

"""
structure.py

Contains base class for representing a protein structure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from typing import Union, Tuple
import gemmi
import re

SUPPORTED_EXTENSIONS = ("pdb", "cif", "mmcif")

# Missing residues (PDB REMARK 465)
# Assume all missing residues are recorded in REMARK 465
_REMARK465_LINE = re.compile(
    r"^REMARK\s+465\s+"
    r"(?P<resname>[A-Z]{3})\s+"
    r"(?P<chain>[A-Za-z0-9])\s*"
    r"(?P<resseq>-?\d+)"
    r"(?:\s+(?P<icode>[A-Za-z]))?\s*$"
)


# Amino acid 1-letter to 3-letter code mapping
AA1_TO_3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


class ProteinStructure:
    """Base class for representing protein structures

    Standardized class for representing protein structures.

    Attributes:
        structure_cif: Internal CIF format of the structure (converted from PDB if needed)
        b_factor_type: What the B-factor column contains (default is UNSPECIFIED)
    """

    def __init__(
        self, 
        structure_filepath_or_content: Union[Path, str, "ProteinStructure"],
        validate_missing_residues: bool = True,
        strict_missing_residues: bool = True,
        check_internal_gaps: bool = True,
    ) -> None:

        # define these placeholders first
        self._gemmi_struct: Optional[gemmi.Structure] = None
        self._backbone_struct: Optional[gemmi.Structure] = None

        # Missing residues parsed from PDB REMARK 465 (if present)
        # - _missing_residues_triplets: {chain: [(resseq, icode, resname), ...]}
        # - _missing_residues: {chain: [resseq, ...]} (sorted unique)
        self._missing_residues_triplets: Dict[str, List[Tuple[int, str, str]]] = {}
        self._missing_residues: Dict[str, List[int]] = {}
        self._missing_residues_validation_report: Dict[str, Any] = {}

        # If passed a ProteinStructure object, use it directly
        if isinstance(structure_filepath_or_content, ProteinStructure):
            self.structure = structure_filepath_or_content.structure
            self.structure_format = structure_filepath_or_content.structure_format
            self._gemmi_struct = structure_filepath_or_content._gemmi_struct
            self._backbone_struct = structure_filepath_or_content._backbone_struct
            self._missing_residues_triplets = (
                structure_filepath_or_content._missing_residues_triplets
            )
            self._missing_residues = structure_filepath_or_content._missing_residues
            self._missing_residues_validation_report = (
                structure_filepath_or_content._missing_residues_validation_report
            )
            return

        # Initialize the structure content and format strings
        structure_content = structure_filepath_or_content
        structure_format = None

        # If a file path is provided, load the structure content and format
        if str(structure_filepath_or_content).lower().endswith(SUPPORTED_EXTENSIONS):
            structure_content = load_structure_file(structure_filepath_or_content)

        # Validate the structure
        if not is_valid_structure(structure_filepath_or_content=structure_content):
            raise ValueError("Structure content is invalid")

        # Otherwise, detect the structure format from the content string
        structure_format = detect_structure_format(structure_content)

        # Save the structure content and format
        self.structure = structure_content
        self.structure_format = structure_format

        # Missing residue validation report is reset/filled by _init_missing_residues_from_structure().
        # Parse and validate missing residues from PDB REMARK 465 (if applicable)
        if validate_missing_residues:
            self._init_missing_residues_from_structure(
                strict=strict_missing_residues,
                check_internal_gaps=check_internal_gaps,
            )

    def _init_missing_residues_from_structure(
        self,
        *,
        strict: bool,
        check_internal_gaps: bool,
    ) -> None:
        """Parse REMARK 465 missing residues (PDB only) and validate against coordinates."""
        # Always reset the last validation report.
        self._missing_residues_validation_report = {}
        # Only PDB text can contain REMARK 465.
        if self.structure_format != "pdb":
            self._missing_residues_triplets = {}
            self._missing_residues = {}
            return

        triplets = extract_missing_residues_from_pdb_text(
            self.structure, return_triplets=True
        )
        self._missing_residues_triplets = triplets
        self._missing_residues = {
            ch: sorted({r for (r, _ic, _rn) in items}) for ch, items in triplets.items()
        }

        # If there are no REMARK 465 entries, nothing to validate.
        if not self._missing_residues_triplets:
            return

        report = validate_missing_residues_against_coordinates(
            self.structure,
            self._missing_residues_triplets,
            check_internal_gaps=check_internal_gaps,
        )
        self._missing_residues_validation_report = report

        conflicts = report.get("conflicts_present_in_atoms", [])
        implied = report.get("implied_internal_gaps_not_in_remark", [])

        if strict and (conflicts or implied):
            parts: List[str] = []
            if conflicts:
                parts.append(
                    "REMARK 465 lists residues as missing, but ATOM/HETATM records exist for: "
                    + ", ".join(
                        [f"{c}:{r}{ic or ''}({rn})" for (c, r, ic, rn) in conflicts]
                    )
                )
            if check_internal_gaps and implied:
                preview = ", ".join([f"{c}:{r}" for (c, r) in implied[:50]])
                parts.append(
                    "Internal residue-number gaps found in ATOM/HETATM not covered by REMARK 465: "
                    + preview
                    + (" ..." if len(implied) > 50 else "")
                )
            raise ValueError("\n".join(parts))

    @property
    def gemmi_struct(self) -> gemmi.Structure:
        """
        Lazy loads the gemmi structure from the internal structure representation.

        Returns:
            gemmi.Structure: The parsed structure object
        """
        if self._gemmi_struct is None:
            if self.structure_format == "cif":
                doc = gemmi.cif.read_string(self.structure)

                # Find first valid structure block
                for block in doc:
                    struct = gemmi.make_structure_from_block(block)
                    if struct is not None and len(struct) > 0:
                        self._gemmi_struct = struct
                        break

                if self._gemmi_struct is None:
                    raise ValueError("No valid structure found in CIF content")
            else:
                self._gemmi_struct = gemmi.read_pdb_string(self.structure)

        return self._gemmi_struct

    @property
    def minimal_backbone_structure(self) -> gemmi.Structure:
        """
        Get the minimal-backbone structure (N, CA, C, O only).
        Lazily builds it if needed.
        """
        if self._backbone_struct is None:
            self._backbone_struct = make_minimal_backbone_structure(self.gemmi_struct)
        return self._backbone_struct

    def get_overridden_sequence_backbone_pdb(self, sequence_override_str: str) -> str:
        """
        Get the PDB string of the minimal-backbone structure with the sequence overridden.
        """
        return pdb_sequence_override(
            self.minimal_backbone_structure, sequence_override_str
        ).make_pdb_string()

    @property
    def structure_pdb(self) -> str:
        """Converts the CIF representation of the structure to a PDB string."""
        if self.structure_format == "cif":
            return convert_cif_str_to_pdb_str(self.structure)
        else:
            return self.structure

    @property
    def structure_cif(self) -> str:
        """Converts the PDB representation of the structure to a CIF string."""
        if self.structure_format == "pdb":
            return convert_pdb_str_to_cif_str(self.structure)
        else:
            return self.structure
        
    # ===============================
    # Missing residues (PDB REMARK 465)
    # ===============================
    @property
    def missing_residues(self) -> Dict[str, List[int]]:
        """Missing residue positions parsed from PDB REMARK 465.

        Returns:
            Dict mapping chain ID to sorted unique residue numbers (resseq).
        """
        return self._missing_residues

    @property
    def missing_residues_triplets(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """Missing residues as (resseq, icode, resname) triplets parsed from REMARK 465."""
        return self._missing_residues_triplets

    @property
    def missing_residues_validation_report(self) -> Dict[str, Any]:
        """Validation report for REMARK 465 parsing (empty if not validated / not applicable)."""
        return self._missing_residues_validation_report

    # ===============================
    # File I/O
    # ===============================
    def write_cif(self, filepath: Union[Path, str]) -> None:
        """
        Write the structure to a CIF file.

        Args:
            filepath: Path where to save the CIF file
        """
        Path(filepath).write_text(self.structure_cif)

    def write_pdb(self, filepath: Union[Path, str]) -> None:
        """
        Write the structure to a PDB file.

        WARNING: PDB format has limitations that may cause data loss.

        Args:
            filepath: Path where to save the PDB file
        """
        Path(filepath).write_text(self.structure_pdb)

    # ===============================
    # Chain Related
    # ===============================
    def get_chain_sequence(self, chain_id: Optional[str] = None) -> str:
        """
        Extract the sequence of a specific chain from the structure.

        Args:
            chain_id: Chain ID to extract (e.g., 'A'). If None, returns the first chain.

        Returns:
            str: One-letter amino acid sequence of the chain

        Raises:
            ValueError: If specified chain_id is not found or no chains exist

        Examples:
            >>> protein.get_chain_sequence()  # First chain
            'MVLSEGEWQ'
            >>> protein.get_chain_sequence('A')  # Chain A specifically
            'MVLSEGEWQ'
        """
        sequences = self.get_chain_sequences()

        if not sequences:
            raise ValueError("No protein chains found in structure")

        if chain_id is not None:
            if chain_id not in sequences:
                raise ValueError(
                    f"Chain '{chain_id}' not found. Available chains: {list(sequences.keys())}"
                )
            return sequences[chain_id]

        # Return first chain
        return next(iter(sequences.values()))

    def get_chain_sequences(self) -> Dict[str, str]:
        """
        Extract the sequences of all chains in the structure.

        Returns:
            Dict[str, str]: Dictionary mapping chain ID to sequence

        Examples:
            >>> protein.get_chain_sequences()
            {'A': 'MVLSEGEWQ', 'B': 'ACDEFGHIK'}
            >>>
            >>> # Iterate over chains
            >>> for chain_id, sequence in protein.get_chain_sequences().items():
            ...     print(f"Chain {chain_id}: {len(sequence)} residues")
            Chain A: 9 residues
            Chain B: 9 residues
        """
        sequences = {}
        for model in self.gemmi_struct:
            for chain in model:
                polymer = chain.whole()
                if polymer:
                    sequences[chain.name] = polymer.make_one_letter_sequence()
        return sequences

    def get_chain_ids(self) -> List[str]:
        """
        Extract the IDs of all chains in the structure.

        Returns:
            List[str]: List of chain IDs
        """
        return list(self.get_chain_sequences().keys())

    @property
    def num_chains(self) -> int:
        """
        Get the number of residues in the structure.
        """
        return len(self.get_chain_sequences())

    # ===============================
    # Residue Related
    # ===============================

    def get_residue_position_map(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Gets a dictionary mapping chain IDs to lists of tuples of (residue_id, position)
        in the chain. Residue ID is the 1-letter code of the residue.
        """
        position_map = {}
        for model in self.gemmi_struct:
            for chain in model:
                chain_id = chain.name
                position_map[chain_id] = []
                chain_sequence = chain.whole()
                residue_id_list = gemmi.one_letter_code(
                    [residue.name for residue in chain_sequence]
                )
                position_list = [residue.seqid.num for residue in chain_sequence]
                position_map[chain_id] = list(zip(residue_id_list, position_list))
        return position_map

    @property
    def num_residues(self) -> int:
        """
        Get the number of residues in the structure.
        """
        return sum(len(chain) for chain in self.get_chain_sequences().values())


# ===============================
# Backbone Related
# ===============================
def make_minimal_backbone_structure(
    structure: gemmi.Structure,
    atoms: Tuple[str, ...] = ("N", "CA", "C", "O"),
) -> gemmi.Structure:
    """
    Return a NEW gemmi.Structure that keeps only the specified backbone atoms
    (default: N, CA, C, O) for all residues.

    The input structure is not modified.
    """
    new_struct = structure.clone()
    keep = set(atoms)

    for model in new_struct:
        for chain in model:
            for residue in chain:
                # iterate backwards over indices so we can safely delete
                for i in range(len(residue) - 1, -1, -1):
                    atom = residue[i]
                    if atom.name not in keep:
                        # Gemmi residues are list-like; deleting by index is safe
                        del residue[i]

    return new_struct


# ===============================
# Sequence override
# ===============================
def pdb_sequence_override(
    structure: gemmi.Structure,
    sequence_override_str: str,
) -> gemmi.Structure:
    """
    Return a NEW gemmi.Structure with residue types overridden according to a
    comma-separated string like:

        "A15:C, A42:L, A58:M"

    where:
        - 'A'  = chain ID
        - '15' = 1-indexed residue number (gemmi residue.seqid.num)
        - 'C'  = new 1-letter amino acid code (standard 20 AAs)

    The input structure is not modified.
    """
    new_struct = structure.clone()

    # Split "A15:C, A42:L" → ["A15:C", "A42:L"]
    raw_overrides = [s.strip() for s in sequence_override_str.split(",") if s.strip()]
    if not raw_overrides:
        return new_struct

    pattern = re.compile(r"^([A-Za-z])(\d+):([A-Za-z])$")

    for spec in raw_overrides:
        m = pattern.match(spec)
        if not m:
            raise ValueError(f"Invalid override spec '{spec}'. Expected format like 'A15:C'.")
        chain_id, pos_str, new_aa1 = m.groups()
        pos = int(pos_str)
        new_aa1 = new_aa1.upper()

        if new_aa1 not in AA1_TO_3:
            raise ValueError(f"Unsupported amino acid '{new_aa1}' in '{spec}'")

        new_resname = AA1_TO_3[new_aa1]

        # Find chain
        target_chain = None
        for model in new_struct:
            for chain in model:
                if chain.name == chain_id:
                    target_chain = chain
                    break
            if target_chain is not None:
                break

        if target_chain is None:
            raise ValueError(f"Chain '{chain_id}' not found for override '{spec}'")

        # Find residue by seqid.num (1-indexed)
        target_res = None
        for residue in target_chain:
            if residue.seqid.num == pos:
                target_res = residue
                break
 
        # Apply override: just change residue name
        # we should tolerate missing residues for overrides
        if target_res is not None:
            target_res.name = new_resname

    return new_struct

# ===============================
# Missing Residues Related
# ===============================

def _parse_observed_residues_from_atoms(pdb_text: str) -> Dict[str, set]:
    """Parse observed residues from ATOM/HETATM records.
    Returns: {chain: set((resseq:int, icode:str))}
    """
    observed_by_chain: Dict[str, set] = {}
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if len(line) < 27:
            continue

        chain = (line[21] or "").strip()  # PDB col 22
        try:
            resseq = int(line[22:26])      # PDB cols 23-26
        except ValueError:
            continue
        icode = (line[26] or "").strip()  # PDB col 27

        observed_by_chain.setdefault(chain, set()).add((resseq, icode))
    return observed_by_chain


def extract_missing_residues_from_pdb_text(
    pdb_text: str,
    chain: Optional[str] = None,
    return_triplets: bool = False,
) -> Union[Dict[str, List[int]], Dict[str, List[Tuple[int, str, str]]]]:
    """Parse PDB REMARK 465 missing residues."""
    missing: Dict[str, set] = {}

    for line in pdb_text.splitlines():
        if not line.startswith("REMARK 465"):
            continue
        if "MISSING RESIDUES" in line or "END MISSING RESIDUES" in line:
            continue

        m = _REMARK465_LINE.match(line)
        if not m:
            continue

        resname = m.group("resname")
        ch = m.group("chain")
        resseq = int(m.group("resseq"))
        icode = (m.group("icode") or "").strip()

        if chain is not None and ch != chain:
            continue

        key = (resseq, icode, resname) if return_triplets else resseq
        missing.setdefault(ch, set()).add(key)

    if return_triplets:
        return {ch: sorted(items, key=lambda x: (x[0], x[1], x[2])) for ch, items in missing.items()}
    return {ch: sorted(items) for ch, items in missing.items()}


def validate_missing_residues_against_coordinates(
    pdb_text: str,
    missing_by_chain: Dict[str, List[Tuple[int, str, str]]],
    *,
    check_internal_gaps: bool = True,
) -> Dict[str, Any]:
    """Validate REMARK 465 missing residues against ATOM/HETATM records."""
    observed_by_chain = _parse_observed_residues_from_atoms(pdb_text)

    conflicts: List[Tuple[str, int, str, str]] = []
    implied_gaps_not_in_remark: List[Tuple[str, int]] = []

    for chain, miss_list in missing_by_chain.items():
        obs = observed_by_chain.get(chain, set())

        for (resseq, icode, resname) in miss_list:
            if (resseq, icode) in obs:
                conflicts.append((chain, resseq, icode, resname))

        if check_internal_gaps:
            obs_nums = sorted({r for (r, _ic) in obs})
            if obs_nums:
                remark_missing_nums = {r for (r, _ic, _rn) in miss_list}
                for a, b in zip(obs_nums, obs_nums[1:]):
                    if b - a <= 1:
                        continue
                    for r in range(a + 1, b):
                        if r not in remark_missing_nums:
                            implied_gaps_not_in_remark.append((chain, r))

    return {
        "conflicts_present_in_atoms": conflicts,
        "implied_internal_gaps_not_in_remark": implied_gaps_not_in_remark,
    }

def load_structure_file(filepath: Union[Path, str]) -> str:
    """
    Loads the contents of a structure file (PDB or CIF) and returns it as a string.

    Args:
        filepath: Path to the structure file (PDB or CIF).

    Returns:
        String of content from the structure file.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file extension is not .pdb, .cif, or .mmcif
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Normalize extension check
    suffix_lower = filepath.suffix.lower()
    if not suffix_lower.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(
            f"Invalid structure file extension: {filepath.suffix}. "
            f"Must one of the following extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Read the structure file
    return filepath.read_text(encoding="utf-8")


def detect_structure_format(structure_content: str) -> str:
    """
    Detect if structure content is CIF or PDB format.

    Args:
        structure_content: Structure file content as string

    Returns:
        "cif" or "pdb"
    """
    # Strip leading whitespace and get first meaningful lines
    lines = [line.strip() for line in structure_content.split("\n") if line.strip()]

    if not lines:
        raise ValueError("Empty structure content. Must be PDB or CIF format.")

    first_line = lines[0]

    # CIF files start with specific markers
    if first_line.startswith("data_"):
        return "cif"

    # Check for CIF loop_ or category markers in first few lines
    for line in lines[:10]:
        if line.startswith("loop_") or line.startswith("_"):
            return "cif"

    # PDB files typically start with specific record types
    pdb_keywords = ["HEADER", "TITLE", "ATOM", "HETATM", "MODEL", "CRYST1"]
    for line in lines[:20]:
        if any(line.startswith(keyword) for keyword in pdb_keywords):
            return "pdb"

    # If still unsure, check for CIF-style underscore fields anywhere
    if any("_atom_site" in line or "_entity" in line for line in lines[:50]):
        return "cif"

    # Default to PDB if we see ATOM/HETATM anywhere in first 100 lines
    for line in lines[:100]:
        if line.startswith(("ATOM", "HETATM")):
            return "pdb"

    raise ValueError("Could not determine structure format (CIF or PDB).")


def is_valid_structure(structure_filepath_or_content: Union[str, Path]) -> bool:
    """
    Ensures that a structure content string/file has valid PDB or CIF format
    by checking the following:
    - there is at least one model in the structure
    - there is at least one atom in the structure

    Args:
        structure_filepath_or_content (Union[str, Path]): Path to the structure file or string of
            content (PDB or CIF).
    Returns:
        True if the structure content string/file has valid PDB or CIF format, False otherwise
    """
    try:
        # Determine if input is a file path or content string
        input_str = str(structure_filepath_or_content)

        if input_str.lower().endswith(SUPPORTED_EXTENSIONS):
            # It's a file path - read directly
            structure = gemmi.read_structure(input_str)
        else:
            struct_format = detect_structure_format(input_str)
            if struct_format == "cif":
                doc = gemmi.cif.read_string(input_str)
                structure = gemmi.make_structure_from_block(doc[0])
            elif struct_format == "pdb":
                structure = gemmi.read_pdb_string(input_str)

    except Exception:
        # If parsing fails for any reason, return False
        return False

    # Must have at least one atom
    has_atoms = False
    for model in structure:
        for chain in model:
            for residue in chain:
                if len(residue) > 0:  # residue has atoms
                    has_atoms = True
                    break
            if has_atoms:
                break
        if has_atoms:
            break

    if not has_atoms:
        return False

    return True


def convert_pdb_str_to_cif_str(pdb_content: str) -> str:
    """
    Converts a structure from PDB format to mmCIF format using gemmi.

    Args:
        pdb_content: Structure content in PDB format

    Returns:
        Structure in mmCIF format (empty string if input is empty)
    """
    if not pdb_content.strip():
        return ""

    try:
        structure = gemmi.read_pdb_string(pdb_content)
        doc = structure.make_mmcif_document()
        return doc.as_string()
    except Exception as e:
        raise ValueError(f"Failed to convert PDB to CIF: {e}")


def convert_cif_str_to_pdb_str(cif_content: str) -> str:
    """
    Converts a structure from mmCIF format to PDB format using gemmi.

    WARNING: PDB format has limitations that may cause data loss:
    - Chain IDs limited to 1 character (multi-character chains truncated)
    - Coordinate precision limited to 3 decimal places
    - Line length limited to 80 characters
    - Atom serial numbers limited to 99,999
    - Residue numbers limited to 9,999

    Args:
        cif_content: Structure content in mmCIF format

    Returns:
        Structure in PDB format (empty string if input is empty)
    """
    if not cif_content.strip():
        return ""

    try:
        doc = gemmi.cif.read_string(cif_content)

        # Find first valid structure block
        structure = None
        for block in doc:
            try:
                structure = gemmi.make_structure_from_block(block)
                if structure is not None and len(structure) > 0:
                    break
            except Exception:
                continue

        if structure is None:
            raise ValueError("No valid structure found in CIF content")

        # Convert to PDB string using gemmi's make_pdb_string
        return structure.make_pdb_string()

    except Exception as e:
        raise ValueError(f"Failed to convert CIF to PDB: {e}")
