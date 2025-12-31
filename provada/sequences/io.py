"""
io.py

I/O operations for sequence files
"""

import hashlib
from pathlib import Path
from Bio import SeqIO
from typing import List

from provada.utils.log import get_logger

logger = get_logger(__name__)


def get_sequence(file: Path) -> str:
    """
    Get the sequence from a file
    """
    file = Path(file)
    if file.suffix == ".fasta":
        seq = str(SeqIO.read(file, "fasta").seq)
    elif file.suffix == ".txt":
        with open(file, "r") as f:
            seq = f.read().strip()
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}")

    return seq


def hash_sequence(sequence: str) -> str:
    """
    Standardized method for producing a hash for sequences
    """
    return hashlib.sha256(sequence.encode()).hexdigest()


def get_fixed_positions_from_file(file: Path, assume_1_indexed: bool = True) -> List[int]:
    """
    Reads in a list of integers representing fixed positions from a file. Assumes
    the file contains one integer per line.

    NOTE: The convention for protein residue position numbering is to use 1-indexing.
    If assume_1_indexed is True, the fixed indices will be converted to 0-indexing.

    Args:
        file (Path): The path to the file containing the fixed indices.
        assume_1_indexed (bool): Whether to assume the fixed indices are 1-indexed.

    Returns:
        List[int]: The fixed indices.
    """
    file = Path(file)
    if file.suffix == ".txt":
        with open(file, "r") as f:
            fixed_indices = [int(line.strip()) for line in f.readlines()]
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}")

    # Convert to 0-indexing if necessary
    if assume_1_indexed:
        logger.info(
            f"NOTE: Assuming 1-indexing for fixed position file {file}. Converting to 0-indexing."
        )
        fixed_indices = [index - 1 for index in fixed_indices]
    else:
        logger.info(
            f"NOTE: Assuming 0-indexing for fixed position file {file}. No conversion necessary."
        )

    return fixed_indices
