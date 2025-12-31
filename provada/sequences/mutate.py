"""
mutate.py

Contains functions for applying mutations to string sequences
"""

import random
from itertools import combinations, product
from Bio.Data.CodonTable import unambiguous_dna_by_name
from Bio.Data import IUPACData
import itertools
from provada.sequences.vocab import AMINO_ACIDS
from typing import List


def mutate_k(
    sequence: str,
    k: int,
    codon_scheme: str = "UNIFORM",
    fixed_indices: List[int] = None,
) -> str:
    """
    Applies k random mutations to a nucleotide or protein sequences. Returns the
    mutated sequence.

    Args:
        sequence: The sequence to mutate.
        k: The number of mutations to apply.

    Returns:
        The mutated sequence.
    """

    if k > len(sequence):
        raise ValueError(
            f"Requested number of mutations ({k}) is greater than the length of the sequence ({len(sequence)})"
        )

    # Make a copy of the sequence to mutate
    mutated_sequence = list(sequence)

    if fixed_indices is None:
        fixed_indices = []

    # Create a list of maskable indices
    maskable_indices = [i for i in range(len(sequence)) if i not in fixed_indices]

    # Randomly select k positions to mutate
    positions = random.sample(maskable_indices, k)

    # Mutate the selected positions
    for position in positions:

        # Determine which characters can be swapped for the current character
        valid_swaps = set(get_codon_scheme(codon_scheme)["amino_acids"]) - set(
            sequence[position]
        )

        # Select a random character from the valid swaps using distribution defined by the codon scheme
        new_char = ""
        while new_char not in valid_swaps:
            new_char = sample_amino_acid(codon_scheme, allow_stop=False)

        # Replace the current character with the new character
        mutated_sequence[position] = new_char

    # Convert the mutated sequence back to a string
    mutated_sequence = "".join(mutated_sequence)

    return mutated_sequence


def mutate_p(
    sequence: str, p: float, codon_scheme: str = "NNN", fixed_indices: List[int] = None
) -> str:
    """
    Applies random mutations to p% of the sequence. Returns the mutated sequence.

    Args:
        sequence: The sequence to mutate.
        p: The probability of mutating a position.

    Returns:
        The mutated sequence.
    """

    if p > 1 or p < 0:
        raise ValueError(f"Probability of mutation must be between 0 and 1, value: {p}")

    if fixed_indices is None:
        fixed_indices = []

    # Determine how many positions are designable
    num_designable_positions = len(sequence) - len(fixed_indices)

    # Determine the number of mutations to apply
    k = max(1, int(p * num_designable_positions))

    # Mutate the sequence
    mutated_sequence = mutate_k(sequence, k, codon_scheme, fixed_indices)

    return mutated_sequence


def get_all_variants_at_mutational_distance_k(sequence: str, k: int) -> list[str]:
    """
    Returns all possible variants of a sequence at a given mutational distance k.
    Only substitutions are considered (no insertions or deletions), and positions
    are mutated simultaneously (i.e., exactly k positions are changed).

    Args:
        sequence: The original sequence to mutate.
        k: The mutational (Hamming) distance.

    Returns:
        A list of all possible variants of the sequence at the given mutational distance k.
    """
    if k > len(sequence):
        raise ValueError(
            f"Requested number of mutations ({k}) is greater than the length of the sequence ({len(sequence)})"
        )

    variants = set()
    for idx_combo in combinations(range(len(sequence)), k):
        # For each combination of positions, create all combinations of replacement characters
        for replacements in product(AMINO_ACIDS, repeat=k):
            # Skip cases where all replacements are the same as the original sequence
            if all(sequence[i] == r for i, r in zip(idx_combo, replacements)):
                continue
            mutated = list(sequence)
            for i, r in zip(idx_combo, replacements):
                mutated[i] = r
            variants.add("".join(mutated))

    return list(variants)


# ===============================
# Codon-based mutagenesis
# ===============================

# Create codon lookup map utilizing the standard codon table
tbl = unambiguous_dna_by_name["Standard"]
CODON_TO_AA = {**tbl.forward_table, **{c: "*" for c in tbl.stop_codons}}


# Common codon schemes
COMMON_CODON_SCHEMES = ["NNN", "NNK", "NNS", "NDT", "DBK", "NRT", "UNIFORM"]

# Contains IUPAC codes for each character in the codon scheme
CODON_SCHEME_VOCAB = IUPACData.ambiguous_dna_values

# Cache for codon schemes. Automatically populated with common codon schemes.
CODON_SCHEME_CACHE = {}

def get_codon_scheme(codon_scheme: str) -> dict:
    """
    Returns a dictionary containing the codons and amino acids for a given codon
    scheme.
    """

    # Ensure the codon scheme is in uppercase
    codon_scheme = codon_scheme.upper()
    if (
        any(c not in CODON_SCHEME_VOCAB for c in codon_scheme)
        and codon_scheme != "UNIFORM"
    ):
        raise ValueError(
            f"Invalid codon scheme: {codon_scheme}. Some valid options include: {CODON_SCHEME_VOCAB}"
        )

    global CODON_SCHEME_CACHE
    if codon_scheme not in CODON_SCHEME_CACHE:

        if codon_scheme == "UNIFORM":
            # Define a uniform codon scheme where each amino acid is equally likely
            CODON_SCHEME_CACHE[codon_scheme] = {
                "codons": [],
                "amino_acids_with_stop": list(AMINO_ACIDS) + ["*"],
                "amino_acids": list(AMINO_ACIDS),
            }

        else:
            # Collect options for each position in the codon scheme
            position_options = [CODON_SCHEME_VOCAB[c] for c in codon_scheme]

            # Build all possible codons
            codons = list(
                "".join(codon) for codon in itertools.product(*position_options)
            )

            # Cache the codons and the amino acids they encode for this scheme
            CODON_SCHEME_CACHE[codon_scheme] = {
                "codons": codons,
                "amino_acids_with_stop": [CODON_TO_AA[codon] for codon in codons],
                "amino_acids": [
                    CODON_TO_AA[codon] for codon in codons if CODON_TO_AA[codon] != "*"
                ],
            }

    return CODON_SCHEME_CACHE[codon_scheme]


# Automatically populate the cache with common codon schemes
for codon_scheme in COMMON_CODON_SCHEMES:
    _ = get_codon_scheme(codon_scheme)


def sample_amino_acid(codon_scheme: str = "UNIFORM", allow_stop: bool = False) -> str:
    """
    Samples an amino acid utilizing a specified codon scheme. Codon schemes are
    often utilized in random mutagenesis experiments to generate variants of coding
    sequences with a bias against stop codons or in a way that introduces bias towards
    other specific amino acids.

    Read more on commonly utilized codon schemes here:
    https://en.wikipedia.org/wiki/Saturation_mutagenesis#:~:text=Different%20degenerate%20codons

    Args:
        codon_scheme: The codon scheme to use.
        allow_stop: Whether to allow stop codons. If True, stop amino acid positions
            will be returned as "*"

    Returns:
        The sampled amino acid.
    """

    # Get the codon scheme
    codon_scheme = get_codon_scheme(codon_scheme)

    # Sample an amino acid from the codon scheme
    if allow_stop:
        return random.choice(codon_scheme["amino_acids_with_stop"])
    else:
        return random.choice(codon_scheme["amino_acids"])
