import os
import pandas as pd
import random
from itertools import combinations
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from typing import Union
from provada.utils.sequences.pairwise_metrics import (
    sequence_similarity,
    sequence_identity,
)

AA_LIST = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y', '_']  # '_' is used for masking positions


def get_sequence(file):
    with open(file, 'r') as f:
        seq = f.read().strip()
    return seq


def get_csv(filename):
    columns = ["design", "mpnn_score"]

    # Check if file exists
    if not os.path.exists(filename):
        # Create a DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)
        df.to_csv(filename, index=False)
    
    df = pd.read_csv(filename)
    return df


def aa_to_arr(seq: str, default_to_A = True) -> np.ndarray:
    """
    Convert a sequence of amino acids (str) to a numpy array of integers.
    Each amino acid is mapped to its index in AA_LIST.
    '_' is converted to -1.
    
    If default_to_A is True, unknown characters map to index of 'A' (0);
    otherwise a ValueError is raised.
    """
    if isinstance(seq, np.ndarray):
        return seq
    if not isinstance(seq, str):
        raise ValueError("Input must be a string")

    aa2idx = {aa: idx for idx, aa in enumerate(AA_LIST)}
    arr = []
    for aa in seq:
        if aa == '_':
            arr.append(-1)
        elif aa in aa2idx:
            arr.append(aa2idx[aa])
        else:
            if default_to_A:
                arr.append(aa2idx['A'])
            else:
                raise ValueError(f"Invalid character '{aa}' in sequence.")

    return np.array(arr, dtype=int)


def arr_to_aa(arr):
    if isinstance(arr, np.ndarray):
        return ''.join([ AA_LIST[i] for i in arr])
    if isinstance(arr, str):
        return arr


def parse_masked_seq(masked_seq_str: str) -> list:
    """
    Convert a masked sequence string (letters + '_') to a list where
    letters remain and '_' indicates a masked position.
    """
    masked = []
    for char in masked_seq_str:
        if char == '_':
            masked.append('_')
        elif char in AA_LIST:
            masked.append(char)
        else:
            raise ValueError(f"Invalid character '{char}'; use single-letter AAs or '_' for masks.")
    return masked


def masked_seq_arr_to_str(masked_seq: np.ndarray) -> str:
    """
    Convert a masked sequence array (1D numpy array) to a string.
    """
    if isinstance(masked_seq, str):
        return masked_seq
    if not isinstance(masked_seq, np.ndarray):
        raise ValueError("Input must be a numpy array")
    return ''.join([AA_LIST[i] if i>= 0 else '_' for i in masked_seq])


def generate_masked_seq_str(
    input_seq: str,
    num_fixed_positions: int,
    hard_fixed_positions: List[int] = []
):
    """
    Generate a masked sequence with a specified number of fixed positions.
    """
    if input_seq is None:
        raise ValueError("input_seq cannot be None")

    if not isinstance(input_seq, str):
        raise ValueError("input_seq must be a string")
    
    if isinstance(num_fixed_positions, float):
        num_fixed_positions = int(num_fixed_positions)

    designalble_seq_length = len(input_seq) - len(hard_fixed_positions)
    if num_fixed_positions > designalble_seq_length:
        raise ValueError("num_fixed_positions cannot be greater than the designalble sequence length")
    num_mask_positions = designalble_seq_length - num_fixed_positions
    
    # Generate random masked positions from the not hard fixed positions
    designalble_positions = [i + 1 for i in range(len(input_seq)) if (i + 1) not in hard_fixed_positions]

    mask_positions = sorted(random.sample(designalble_positions, num_mask_positions))
    
    # Create the masked sequence string
    masked_seq_str = "".join(['_' if (i + 1) in mask_positions else aa for i, aa in enumerate(input_seq)])
    
    return masked_seq_str


def generate_masked_seqs_str(
    input_seqs: List[str],
    num_fixed_positions: int,
    hard_fixed_positions: List[int] = []
):
    """
    Generate masked sequences for a list of input sequences with a specified number of fixed positions.
    Returns a list of masked sequence strings.
    """
    if isinstance(input_seqs, str):
        return [generate_masked_seq_str(input_seqs, num_fixed_positions, hard_fixed_positions)]

    if not isinstance(input_seqs, list) or not all(isinstance(seq, str) for seq in input_seqs):
        raise ValueError("input_seqs must be a list of strings")

    masked_seqs = []
    for seq in input_seqs:
        masked_seq_str = generate_masked_seq_str(seq, num_fixed_positions, hard_fixed_positions)
        masked_seqs.append(masked_seq_str)

    return masked_seqs


def generate_masked_seq_arr(
    input_seq: str,
    num_fixed_positions: int,
    hard_fixed_positions: List[int] = []
):
    """
    Generate a masked sequence with a specified number of fixed positions,
    and return it as a numpy array.
    """
    masked_seq_str = generate_masked_seq_str(input_seq, num_fixed_positions, hard_fixed_positions)

    # Convert the masked sequence string to a numpy array
    masked_seq_arr = aa_to_arr(masked_seq_str)
    return masked_seq_arr


def get_masked_positions(masked_seq_str: str) -> List[int]:
    """
    Return the zero-based indices of every “_” in the input string.
    """
    return [i for i, c in enumerate(masked_seq_str) if c == '_']


def compute_diversity_metrics(
    seqs: Union[List[str], np.ndarray], distance: str = "hamming"
) -> Dict[str, Any]:
    """
    Compute diversity metrics for a set of sequences.

    Accepts:
        - List[str]: list of equal-length strings
        - np.ndarray: 2D array of shape (N, L), dtype=int or str

    Returns a dict:
      {
        "var_flags": List[int],
        "distances": List[int],
        "avg_dist": float,
        "min_dist": int,
        "max_dist": int
      }
    """
    if isinstance(seqs, list):
        # Validate and convert List[str] to np.ndarray
        if not seqs:
            raise ValueError("`seqs` must contain at least one sequence.")
        L = len(seqs[0])
        if any(len(s) != L for s in seqs):
            raise ValueError("All sequences must have the same length.")
        seq_array = np.array([list(s) for s in seqs], dtype="<U1")
    elif isinstance(seqs, np.ndarray):
        if seqs.ndim != 2:
            raise ValueError(f"`seqs` must be 2D, got shape {seqs.shape}")
        seq_array = seqs
    else:
        raise TypeError("`seqs` must be either List[str] or np.ndarray")

    N, L = seq_array.shape

    # 1) Per-column variability
    var_flags = [
        1 if np.unique(seq_array[:, col_idx]).size > 1 else 0 for col_idx in range(L)
    ]

    # 2) Pairwise Hamming distances
    distances = [
        int(np.sum(seq_array[i] != seq_array[j])) for i, j in combinations(range(N), 2)
    ]

    # 3) Summary stats
    if distances:
        avg_dist = sum(distances) / len(distances)
        min_dist = min(distances)
        max_dist = max(distances)
    else:
        avg_dist = min_dist = max_dist = 0

    return {
        "var_flags": var_flags,
        "distances": distances,
        "avg_dist": avg_dist,
        "min_dist": min_dist,
        "max_dist": max_dist,
    }


def mask_seq_str_to_arr(masked_seq_str):
    """
    Convert a masked sequence string to a numpy array.
    """
    # Define the mapping of amino acids to integers
    aa_to_int = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        '_': -1
    }

    # Convert the sequence string to a list of integers
    int_seq = [aa_to_int[aa] for aa in masked_seq_str]

    return np.array(int_seq)


def append_csv_line(
    path: Path,
    line: Dict[str, Any]
) -> None:
    """
    Append a single row (as a dict) to the CSV at `path` using pandas.
    Writes header if file doesn’t exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([line])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def get_mismatch_fraction_multiseqs(
    seqs: Union[List[str], np.ndarray], ref_seq: str, use_similarity: bool = True
) -> List[float]:
    """
    Compute the mismatch fraction for a list of sequences with respect to a
    reference sequence. If use_similarity is True, we use the sequence_similarity
    (based on a substitution matrix) to compute the mismatch fraction.

    Args:
        seqs (List[str]): List of sequences to compute the mismatch fraction for.
        ref_seq (str): Reference sequence to compute the mismatch fraction with respect to.
        use_similarity (bool): If True, use the sequence_similarity to compute the mismatch fraction.

    Returns:
        List[float]: List of mismatch fractions.
    """

    comparison_function = sequence_similarity if use_similarity else sequence_identity

    # Create dataframe containing the sequences
    mismatch_fracs = [
        1 - comparison_function(arr_to_aa(arr_seq), ref_seq) for arr_seq in seqs
    ]

    return mismatch_fracs


def read_fixed_positions(position_txt):
    """
    Read fixed positions from a text file.
    Each line should contain a single integer representing a position.
    Returns a list of integers.
    """
    if not os.path.exists(position_txt):
        raise FileNotFoundError(f"File {position_txt} does not exist.")
    
    with open(position_txt, 'r') as f:
        positions = [int(line.strip()) for line in f if line.strip().isdigit()]
    
    return positions
