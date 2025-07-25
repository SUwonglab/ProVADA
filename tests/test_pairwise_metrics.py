"""
test_pairwise_metrics.py
"""

import pytest


from provada.utils.sequences.pairwise_metrics import (
    get_pairwise_metric,
    __all__ as pairwise_comparison_functions,
)


TEST_CASES = [
    {
        "description": "Identical sequences",
        "seq1": "MARGARET",
        "seq2": "MARGARET",
        "expected": {
            "levenshtein_distance": 0,
            "levenshtein_ratio": 1.0,
            "sequence_identity": 1.0,
            "sequence_similarity": 1.0,
            "normalized_hamming_distance": 0,
        },
    },
    {
        "description": "Similar substitution",
        "seq1": "MARGARET",
        "seq2": "MARGAKET",
        "expected": {
            "levenshtein_distance": 1,
            "levenshtein_ratio": 0.875,
            "sequence_identity": 0.875,
            "sequence_similarity": 1.0,
            "normalized_hamming_distance": 0.125,
        },
    },
]


@pytest.mark.parametrize("pairwise_metric", pairwise_comparison_functions)
@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=[tc["description"] for tc in TEST_CASES]
)
def test_pairwise_metrics(pairwise_metric, test_case):
    pairwise_metric_func = get_pairwise_metric(pairwise_metric)
    value = pairwise_metric_func(test_case["seq1"], test_case["seq2"])

    assert (
        value == test_case["expected"][pairwise_metric]
    ), f"Test case '{test_case['description']}' failed for {pairwise_metric}: expected {test_case['expected'][pairwise_metric]} but got {value}"