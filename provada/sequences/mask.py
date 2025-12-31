"""
mask.py

Utility functions related to masking sequences.
"""

import random
from typing import List


def mask_k(
    sequence: str, k: int, mask_str: str = "_", fixed_indices: List[int] = None
) -> str:
    """
    Mask k random positions of a sequence.

    Args:
        sequence (str): The sequence to mask.
        k (int): The number of positions to mask.
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.
        fixed_indices (List[int]): The indices of the positions that are fixed and
            should not be masked.
    """

    if k > len(sequence):
        raise ValueError("k cannot be greater than the length of the sequence")

    # Create a list of the sequence
    sequence_list = list(sequence)

    if fixed_indices is None:
        fixed_indices = []

    # Create a list of maskable indices
    maskable_indices = [i for i in range(len(sequence)) if i not in fixed_indices]

    # Randomly select k positions to mask
    positions = random.sample(maskable_indices, k)

    # Mask the selected positions
    for position in positions:
        sequence_list[position] = mask_str

    # Convert the list back to a string
    return "".join(sequence_list)


def mask_p(
    sequence: str, p: float, mask_str: str = "_", fixed_indices: List[int] = None
) -> str:
    """
    Mask a random fraction of positions in a sequence.

    Args:
        sequence (str): The sequence to mask.
        p (float): The fraction of positions to mask.
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.

    Returns:
        str: The masked sequence.
    """

    if p > 1 or p < 0:
        raise ValueError("p must be between 0 and 1")

    if fixed_indices is None:
        fixed_indices = []

    # Determine how many positions are designable
    num_designable_positions = len(sequence) - len(fixed_indices)

    # Determine the number of positions to mask
    k = max(1, int(p * num_designable_positions))

    # Mask the sequence
    masked_sequence = mask_k(sequence, k, mask_str, fixed_indices)

    return masked_sequence


def mask_assigned_positions(sequence: str, inds_to_mask: list[int], mask_str: str = "_") -> str:
    """
    Returns a masked version of the sequence where the positions in inds_to_mask
    are replaced with the mask_str.

    Args:
        sequence (str): The sequence to mask.
        inds_to_mask (list[int]): The indices of the positions to mask. (0-indexed)
        mask_str (str): The string of characters that replace sequence characters
            in masked positions.

    Returns:
        str: The masked sequence.
    """

    # Create a list of the sequence
    sequence_list = list(sequence)

    # Mask the assigned positions
    for ind in inds_to_mask:
        sequence_list[ind] = mask_str

    # Convert the list back to a string
    return "".join(sequence_list)
