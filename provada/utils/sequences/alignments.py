"""
alignments.py

This file contains functions related to sequence alignments.
"""
import numpy as np
import biotite.sequence as seq
import biotite.sequence.align as align


def global_pairwise_alignment(seq1: str, seq2: str) -> align.Alignment:
    """
    Returns the best scoring pairwise alignment between two sequences.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        align.Alignment: The best scoring pairwise alignment between the two sequences.
    """
    sequence1 = seq.ProteinSequence(seq1)
    sequence2 = seq.ProteinSequence(seq2)
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignments = align.align_optimal(
        sequence1, sequence2, matrix, gap_penalty=(-10, -0.5)
    )
    return alignments[0]


def dummy_alignment(seq1: str, seq2: str) -> align.Alignment:
    """
    Returns a dummy alignment between two sequences of equal length by creating
    a 1-to-1 correspondence between the positions of the sequences.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        align.Alignment: A dummy alignment between the two sequences.
    """
    length = len(seq1)
    if length != len(seq2):
        raise ValueError("Sequences must be of equal length for dummy_alignment.")

    # Construct the biotite Alignment object with a one to one correspondence for
    # the trace and a dummy alignment score of 0
    alignment = align.Alignment(
        sequences=[seq.ProteinSequence(seq1), seq.ProteinSequence(seq2)],
        trace=np.column_stack((np.arange(length), np.arange(length))),
        score=0,
    )

    return alignment