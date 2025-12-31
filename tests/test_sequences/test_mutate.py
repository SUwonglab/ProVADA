"""
test_mutate.py

"""

import pytest
from provada.sequences.mutate import get_codon_scheme, mutate_k, mutate_p

CODON_SCHEME_TOTAL_CODONS = {
    "NNN": {
        "n_codons": 64,
        "n_stops": 3,
        "n_unique_amino_acids": 20,
    },
    "NNK": {
        "n_codons": 32,
        "n_stops": 1,
        "n_unique_amino_acids": 20,
    },
    "NNS": {
        "n_codons": 32,
        "n_stops": 1,
        "n_unique_amino_acids": 20,
    },
    "NDT": {
        "n_codons": 12,
        "n_stops": 0,
        "n_unique_amino_acids": 12,
    },
    "DBK": {
        "n_codons": 18,
        "n_stops": 0,
        "n_unique_amino_acids": 12,
    },
    "NRT": {
        "n_codons": 8,
        "n_stops": 0,
        "n_unique_amino_acids": 8,
    },
}


@pytest.mark.parametrize("codon_scheme", list(CODON_SCHEME_TOTAL_CODONS.keys()))
def test_codon_scheme_total_codons(codon_scheme):
    codon_scheme_dict = get_codon_scheme(codon_scheme)
    assert (
        len(codon_scheme_dict["codons"])
        == CODON_SCHEME_TOTAL_CODONS[codon_scheme]["n_codons"]
    )
    assert (
        len(codon_scheme_dict["amino_acids_with_stop"])
        - len(codon_scheme_dict["amino_acids"])
        == CODON_SCHEME_TOTAL_CODONS[codon_scheme]["n_stops"]
    )
    assert (
        len(set(codon_scheme_dict["amino_acids"]))
        == CODON_SCHEME_TOTAL_CODONS[codon_scheme]["n_unique_amino_acids"]
    )


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_mutate_k(k):
    sequence = "TARGET"
    masked_sequence = mutate_k(sequence, k, codon_scheme="NNN", fixed_indices=[0])
    assert len(masked_sequence) == len(
        sequence
    ), "Masked sequence length should be the same as the original sequence length"
    assert masked_sequence[0] == "T", "Fixed position should not be changed"


@pytest.mark.parametrize("p", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
def test_mutate_p(p):
    sequence = "TARGET"
    masked_sequence = mutate_p(sequence, p, codon_scheme="NNN", fixed_indices=[0])
    assert masked_sequence[0] == "T", "Fixed position should not be masked"


@pytest.mark.parametrize("fixed_indices", [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
def test_mutate_p_fixed_indices(fixed_indices):
    sequence = "TARGET"
    mutatep_seq = mutate_p(
        sequence, 0.5, codon_scheme="NNN", fixed_indices=fixed_indices
    )
    mutatek_seq = mutate_k(sequence, 1, codon_scheme="NNN", fixed_indices=fixed_indices)

    for ind in fixed_indices:
        assert (
            mutatep_seq[ind] == sequence[ind]
        ), f"Fixed position {ind} should not be mutated"
        assert (
            mutatek_seq[ind] == sequence[ind]
        ), f"Fixed position {ind} should not be mutated"
