"""
pairwise_metrics.py

This file contains functions that compare sequences.
"""

import sys
from scipy.spatial import distance as scipy_distance
from Levenshtein import distance, ratio
from provada.utils.sequences.alignments import global_pairwise_alignment, dummy_alignment
from biotite.sequence import ProteinSequence
import biotite.sequence.align as biotite_align
from typing import Callable

__all__ = [
    "levenshtein_distance",
    "levenshtein_ratio",
    "sequence_identity",
    "sequence_similarity",
    "normalized_hamming_distance",
]


def get_pairwise_metric(metric_name: str) -> Callable:
    """
    Dynamically retrieves a scoring function from the current module.
    """
    # Get the current module object from sys.modules
    current_module = sys.modules[__name__]

    # First, validate that the requested metric is in our public API list
    if metric_name not in __all__:
        raise ValueError(
            f"Invalid pairwise metric: {metric_name}. Select from {__all__}"
        )

    try:
        # Retrieve the function attribute from the module by its string name
        func = getattr(current_module, metric_name)

        # Ensure the retrieved attribute is a callable function
        if not callable(func):
            raise AttributeError  # Treat non-callables as if they don't exist

        return func
    except AttributeError:
        # This handles cases where the function name is in __all__ but not
        # actually defined in the module, or is not a function.
        raise ValueError(
            f"Invalid pairwise metric: {metric_name}. Select from {__all__}"
        )


def more_similar_is_larger(more_similar_is_larger: bool):
    """
    A decorator that adds a 'more_similar_is_larger' attribute to the pairwise
    similarity functions.

    Args:
        more_similar_is_larger (bool): If True, indicates the metric returns a
            larger value for more similar sequences. If False, indicates the
            metric returns a larger value for less similar sequences.
    """

    def decorator(func):
        # Assign the 'more_similar_is_larger' attribute with the given value to the function
        setattr(func, "more_similar_is_larger", more_similar_is_larger)
        return func

    return decorator


@more_similar_is_larger(False)
def levenshtein_distance(seq1: str, seq2: str) -> int:
    """
    Returns the Levenshtein distance between two sequences, which is defined as
    the minimum number of single-character edits (insertions, deletions, or
    substitutions) required to transform one sequence into the other.

    The distance is calculated as the number of non-matching positions in the
    alignment of the two sequences.

    The distance is 0 if the sequences are identical.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        int: The Levenshtein distance between the two sequences.
    """
    return distance(seq1, seq2)


@more_similar_is_larger(True)
def levenshtein_ratio(seq1: str, seq2: str) -> float:
    """
    Returns the Levenshtein ratio between two sequences.

    Calculates a normalized indel similarity in the range [0, 1]. The indel distance calculates the minimum number of insertions and deletions required to change one sequence into the other.

    This is calculated as 1 - (distance / (len1 + len2))

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        float: The Levenshtein ratio between the two sequences.
    """
    return ratio(seq1, seq2)


@more_similar_is_larger(True)
def sequence_identity(
    seq1: str,
    seq2: str,
    skip_alignment_if_same_length: bool = True,
    mode: str = "not_terminal",
) -> float:
    """
    Returns the sequence identity between two sequences. Sequence identity is
    defined as the percentage of identical residues between two sequences in
    aligned positions divided by the "length of the alignment" as defined by the
    `mode` parameter.

    Modes: (default: "not_terminal")
        - "all": Use the total number of columns in the alignment trace
        - "not_terminal": Use the number of columns in the alignment trace
            excluding terminal gaps (i.e. gaps at the begining or end of the
            alignment)
        - "shortest": Use the length of the shortest sequence (i.e. the number of
            columns in the alignment trace)

    Ranges between 0 and 1.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        skip_alignment_if_same_length (bool): If True, the alignment is skipped
            if the sequences are of the same length to decrease computational
            complexity.
        mode (str): Determines the 'length' of the alignment as described above.

    Returns:
        float: The sequence identity between the two sequences.
    """

    if skip_alignment_if_same_length and len(seq1) == len(seq2):
        alignment = dummy_alignment(seq1, seq2)
    else:
        alignment = global_pairwise_alignment(seq1, seq2)

    return biotite_align.get_sequence_identity(alignment, mode=mode)


@more_similar_is_larger(True)
def sequence_similarity(
    seq1: str,
    seq2: str,
    skip_alignment_if_same_length: bool = True,
    mode: str = "not_terminal",
):
    """
    Calculate the sequence similarity for two sequences based on positive
    substitution scores (using BLOSUM62). This implementation is based on biotite's
    get_sequence_identity function.

    The similarity is equal to the number of aligned positions with a
    positive substitution score divided by a measure for the length of
    the alignment that depends on the `mode` parameter.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        skip_alignment_if_same_length (bool): If True, the alignment is skipped
            if the sequences are of the same length to decrease computational
            complexity.
        mode (str): Determines the 'length' of the alignment as described above.

    Returns:
        float: The sequence similarity between the two sequences.
    """

    # Use a standard protein substitution matrix (BLOSUM62)
    matrix = biotite_align.SubstitutionMatrix.std_protein_matrix()

    # Get the best alignment (highest score)
    if skip_alignment_if_same_length and len(seq1) == len(seq2):
        alignment = dummy_alignment(seq1, seq2)
    else:
        alignment = global_pairwise_alignment(seq1, seq2)

    # Convert the sequences to Biotite Sequence objects
    seq1 = ProteinSequence(seq1)
    seq2 = ProteinSequence(seq2)

    # Count Similar Pairs
    similar_pairs = 0
    trace = alignment.trace  # Shape (alignment_length, 2)

    # Iterate over each column in the alignment trace
    for i in range(trace.shape[0]):
        # Get the indices in the original sequences for this column
        idx1 = trace[i, 0]
        idx2 = trace[i, 1]

        # Check if both positions are non-gap
        if idx1 != -1 and idx2 != -1:
            # Get the symbols (amino acids) from the original sequences
            symbol1 = seq1[idx1]
            symbol2 = seq2[idx2]

            try:
                # Look up the score in the substitution matrix
                score = matrix.get_score(symbol1, symbol2)
                # Increment count if the score is positive
                if score > 0:
                    similar_pairs += 1
            except KeyError:
                pass

    # Calculate Effective Alignment Length
    if mode == "all":
        # Use the total number of columns in the alignment trace
        length = trace.shape[0]
    elif mode == "not_terminal":
        start, stop = biotite_align.find_terminal_gaps(alignment)
        if stop <= start:
            # This happens if alignment contains only gaps or sequences do not overlap
            # (less likely in global alignment unless sequences are very different/short)
            raise ValueError(
                "Cannot calculate non-terminal similarity, "
                "sequences have no overlap after alignment"
            )
        length = stop - start
    elif mode == "shortest":
        length = min(len(seq1), len(seq2))
    else:
        raise ValueError(f"'{mode}' is an invalid calculation mode")

    # Calculate and Return Similarity
    if length == 0:
        if similar_pairs > 0:
            raise RuntimeError(
                "Calculated similar pairs > 0 but alignment length is 0."
            )
        return 0.0

    return similar_pairs / length


@more_similar_is_larger(False)
def normalized_hamming_distance(
    seq1: str, seq2: str, skip_alignment_if_same_length: bool = True
) -> int:
    """
    Returns the normalized Hamming distance between two sequences. The Hamming distance is
    defined as the number of positions at which the corresponding elements are
    different. It is generally defined for sequences of equal length.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        skip_alignment_if_same_length (bool): If True, the alignment is skipped
            if the sequences are of the same length to decrease computational
            complexity.

    Returns:
        int: The Hamming distance between the two sequences.
    """
    if skip_alignment_if_same_length and len(seq1) == len(seq2):
        alignment = dummy_alignment(seq1, seq2)
    else:
        alignment = global_pairwise_alignment(seq1, seq2)

    gapped_sequences = alignment.get_gapped_sequences()

    return scipy_distance.hamming(list(gapped_sequences[0]), list(gapped_sequences[1]))