"""
io.py

I/O operations for sequence files
"""

import hashlib
import re
from pathlib import Path
from Bio import SeqIO
from typing import List, Union, Dict, Any, Optional

from provada.utils.log import get_logger

logger = get_logger(__name__)


def get_sequence(file: Path, chain_ids: Optional[List[str]] = None) -> str:
    """
    Get the sequence from a file.
    
    If the file is a .txt file, it is assumed to be a single sequence.
    If the file is a .fasta file:
      - If it contains a single record, the sequence is returned.
      - If it contains multiple records, the sequences are joined by '/', 
        ordered by chain_ids when supplied. Headers must identify those chains.
        Without a structure-defined order, records retain the legacy ID sort.
      - Text and single-record sequences use the supplied structure order for
        their '/'-separated segments.
    """
    file = Path(file)
    if file.suffix == ".fasta":
        records = list(SeqIO.parse(file, "fasta"))
        if not records:
             raise ValueError(f"No sequences found in {file}")
        
        if len(records) == 1:
            seq = str(records[0].seq)
        else:
            by_chain = {record.id: record for record in records}
            if len(by_chain) != len(records):
                raise ValueError("Multichain FASTA must have unique chain IDs")
            if chain_ids is not None:
                if set(by_chain) != set(chain_ids):
                    raise ValueError(f"FASTA chains {sorted(by_chain)} do not match structure chains {chain_ids}")
                records = [by_chain[chain] for chain in chain_ids]
            else:
                records.sort(key=lambda record: record.id)
            seq = "/".join(str(r.seq) for r in records)
            
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


def get_fixed_positions_from_file(file: Path, assume_1_indexed: bool = True) -> Union[List[int], Dict[str, Any]]:
    """
    Reads in fixed positions from a file. 
    Supports:
    1. List of integers (single chain or flattened). Returns List[int].
    2. List of Chain+Position (e.g. "A10") or Chain (e.g. "B" for whole chain). Returns Dict.
       - A10 -> {'A': [9]} (if 1-indexed)
       - B -> {'B': 'ALL'}

    Args:
        file (Path): The path to the file.
        assume_1_indexed (bool): Whether to assume indices are 1-indexed (applies to integers).

    Returns:
        List[int] or Dict[str, Union[List[int], str]]: Fixed positions. 
    """
    file = Path(file)
    content = file.read_text()
    
    # Split by comma or whitespace/newlines
    tokens = [t.strip() for t in re.split(r'[,\s]+', content) if t.strip()]
    
    if not tokens:
        return []

    # Check for simple integers (Legacy/Single Chain)
    if all(re.match(r'^-?\d+$', t) for t in tokens):
         indices = [int(t) for t in tokens]
         if assume_1_indexed:
             logger.info(f"NOTE: Assuming 1-indexing for fixed position file {file}. Converting to 0-indexing.")
             indices = [index - 1 for index in indices]
         else:
             logger.info(f"NOTE: Assuming 0-indexing for fixed position file {file}.")
         return indices
         
    # Multi-chain parsing
    fixed_dict = {}
    for token in tokens:
        # Match Chain+Pos (e.g. A10, A-5)
        # Note: Chain ID is usually 1 char in PDB, but we accept [A-Za-z0-9]+ 
        # provided it ends with digits for the position part.
        
        match_pos = re.match(r"^([A-Za-z0-9]+?)(-?\d+)$", token)
        match_chain = re.match(r"^([A-Za-z0-9]+)$", token)
        
        if match_pos:
            chain, pos = match_pos.groups()
            idx = int(pos)
            if assume_1_indexed:
                idx -= 1
            if chain not in fixed_dict:
                fixed_dict[chain] = []
            
            # If we already have 'ALL' for this chain, ignore specific pos
            if fixed_dict[chain] != "ALL":
                fixed_dict[chain].append(idx)
                
        elif match_chain:
            chain = match_chain.group(1)
            fixed_dict[chain] = "ALL"
            
        else:
             raise ValueError(f"Invalid fixed position token: '{token}'. Must be integer, Chain+Pos (e.g. A10), or Chain (e.g. A).")
             
    return fixed_dict
