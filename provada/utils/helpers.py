import os
import pandas as pd
import random
from itertools import combinations
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from typing import Union


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



def compute_diversity_metrics_str(seqs: List[str]) -> Dict[str, Any]:
    """
    Given a list of equal-length sequences, compute:
      1) var_flags: a list of 0/1 per column (1 if at least one seq differs at that position)
      2) pairwise Hamming distances between all unordered pairs
      3) summary metrics: avg_dist, min_dist, max_dist

    Returns a dict:
      {
        "var_flags": List[int],
        "distances": List[int],
        "avg_dist": float,
        "min_dist": int,
        "max_dist": int
      }
    """
    if not seqs:
        raise ValueError("`seqs` must contain at least one sequence.")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("All sequences must have the same length.")

    # Per-position variability
    var_flags = [
        1 if len(set(col)) > 1 else 0
        for col in zip(*seqs)
    ]

    # Pairwise Hamming
    def hamming(a: str, b: str) -> int:
        return sum(x != y for x, y in zip(a, b))

    pairs = combinations(seqs, 2)
    distances = [hamming(a, b) for a, b in pairs]

    if distances:
        avg_dist = sum(distances) / len(distances)
        min_dist = min(distances)
        max_dist = max(distances)
    else:
        # Only one sequence, no pairs
        avg_dist = min_dist = max_dist = 0

    return {
        "var_flags": var_flags,
        "distances": distances,
        "avg_dist": avg_dist,
        "min_dist": min_dist,
        "max_dist": max_dist,
    }




def compute_diversity_metrics_int(
    seq_array: np.ndarray
) -> Dict[str, Any]:
    """
    Given a 2D numpy array of shape (N, L), where each row is a sequence
    (either dtype=int or dtype='<U1'), compute:

      1) var_flags: List[int] of length L, where 1 indicates at least one
         difference in that column across rows.
      2) pairwise Hamming distances for all unordered row-pairs.
      3) summary metrics: avg_dist, min_dist, max_dist.

    Returns:
      {
        "var_flags": List[int],
        "distances": List[int],
        "avg_dist": float,
        "min_dist": int,
        "max_dist": int
      }
    """
    # 0) basic validation
    if seq_array.ndim != 2:
        raise ValueError(f"`seq_array` must be 2D, got shape {seq_array.shape}")
    N, L = seq_array.shape

    # 1) per-column variability flags
    var_flags = []
    for col_idx in range(L):
        column = seq_array[:, col_idx]
        # np.unique works for both ints and one-char strings
        uniq = np.unique(column)
        var_flags.append(1 if uniq.size > 1 else 0)

    # 2) pairwise Hamming distances
    distances = []
    for i, j in combinations(range(N), 2):
        # boolean array where entries differ, sum → Hamming distance
        dist = int(np.sum(seq_array[i] != seq_array[j]))
        distances.append(dist)

    # 3) summary metrics
    if distances:
        avg_dist = sum(distances) / len(distances)
        min_dist = min(distances)
        max_dist = max(distances)
    else:
        # only one sequence → trivially zero diversity
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



def get_mismatch_fraction_multiseqs(seqs: Union[List[str], np.ndarray], 
                                    ref_seq: Union[str, np.ndarray]) -> List[float]:

    mismatch_fracs = []
    for seq in seqs:
        if len(seq) != len(ref_seq):
            raise ValueError("All sequences must have the same length as the reference sequence.")

        # convert sequences to numpy arrays if they are strings
        if isinstance(seq, str):
            seq = aa_to_arr(seq)
        if isinstance(ref_seq, str):
            ref_seq_arr = aa_to_arr(ref_seq)
        mismatch_frac = int(np.sum(seq != ref_seq_arr)) / len(ref_seq_arr)
        mismatch_fracs.append(mismatch_frac)

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



















