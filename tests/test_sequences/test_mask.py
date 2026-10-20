"""
test_mask.py

Tests for masking functions in provada.sequences.mask
"""

import pytest
from provada.sequences.mask import mask_k, mask_p, number_of_masked_positions


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_mask_k(k):
    sequence = "TARGET"
    masked_sequence = mask_k(sequence, k, mask_str="_", fixed_indices=[0])
    assert len(masked_sequence) == len(
        sequence
    ), "Masked sequence length should be the same as the original sequence length"
    assert (
        sum(c == "_" for c in masked_sequence) == k
    ), "Number of masked positions should be equal to k"
    assert masked_sequence[0] == "T", "Fixed position should not be masked"


@pytest.mark.parametrize("p", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
def test_mask_p(p):
    sequence = "TARGET"
    masked_sequence = mask_p(sequence, p, mask_str="_", fixed_indices=[0])
    assert len(masked_sequence) == len(
        sequence
    ), "Masked sequence length should be the same as the original sequence length"
    assert masked_sequence[0] == "T", "Fixed position should not be masked"


@pytest.mark.parametrize("fixed_indices", [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
def test_mask_p_fixed_indices(fixed_indices):
    sequence = "TARGET"
    maskp_seq = mask_p(sequence, 0.5, mask_str="_", fixed_indices=fixed_indices)
    maskk_seq = mask_k(sequence, 1, mask_str="_", fixed_indices=fixed_indices)

    for ind in fixed_indices:
        assert maskp_seq[ind] != "_", f"Fixed position {ind} should not be masked"
        assert maskk_seq[ind] != "_", f"Fixed position {ind} should not be masked"


@pytest.mark.parametrize(
    "p, num_designable, expected",
    [
        (0.5, 100, 50),
        (0.3, 10, 3),
        # A fraction that rounds down to nothing still masks one position,
        # since a prompt with nothing masked has nothing to design.
        (0.001, 100, 1),
        (0.0, 100, 1),
    ],
)
def test_number_of_masked_positions(p, num_designable, expected):
    assert number_of_masked_positions(p, num_designable) == expected
