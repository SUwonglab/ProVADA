import pandas as pd
import pytest

from provada.components.evaluator import (
    EVALUATOR_REGISTRY,
    get_evaluators_from_score_keys,
    get_evaluator,
    clear_all_evaluator_cache,
    EVALUATOR_INSTANCE_CACHE,
)

from provada.sequences.vocab import GFP, AAV_WT_VP3
from provada.sequences.mutate import mutate_k
from provada.utils.setup import seed_everything
from provada.sequences.io import hash_sequence, get_fixed_positions_from_file


def test_evaluator_selection():
    """
    Tests the get_evaluator_from_scores function
    """

    # Create a list of score keys
    score_keys = ["dummy_score"]

    # Get the evaluators
    evaluators = get_evaluators_from_score_keys(score_keys)

    # Ensure that only one evaluator is returned
    assert len(evaluators) == 1
    assert evaluators[0].__name__ == "Dummy"


def test_result_caching():
    seed_everything(42)

    # Create a dummy evaluator
    evaluator = EVALUATOR_REGISTRY.get_instance("dummy")

    sequences = [GFP] * 10

    mutated_sequences = [mutate_k(sequence, k=10) for sequence in sequences]

    # Run the evaluator on the sequences
    scored_sequences = evaluator.evaluate(sequences=mutated_sequences)

    # Ensure the output order is the same as the input order
    for i, score_dict in enumerate(scored_sequences.to_dict(orient="records")):
        assert score_dict["sequence"] == mutated_sequences[i]

    # Create new mutated sequences
    new_mutated_sequences = [mutate_k(sequence, k=10) for sequence in sequences]

    # Create all sequences (with some duplicates)
    all_sequences = mutated_sequences + new_mutated_sequences + new_mutated_sequences

    all_sequences_df = pd.DataFrame({"sequence": all_sequences})
    all_sequences_df["sequence_hash"] = all_sequences_df["sequence"].map(hash_sequence)

    # Ensure the caching functionality works as expected
    seq_df = evaluator.pre_eval(seq_df=all_sequences_df)

    # seq_df should have 10 old sequences and 20 new sequences
    assert len(seq_df) == 30
    assert (
        seq_df["new_sequence"].sum() == 20
    ), f"Expected 20 new sequences but got {seq_df['new_sequence'].sum()}"

    # The unique new sequences should be the same as the unique new mutated sequences
    assert set(seq_df["sequence"][seq_df["new_sequence"]].unique().tolist()) == set(
        new_mutated_sequences
    ), f"Expected {set(new_mutated_sequences)} but got {set(seq_df['sequence'][seq_df['new_sequence']].unique().tolist())}"

    # Run the evaluator on all the sequences
    all_sequences_scored = evaluator.evaluate(sequences=all_sequences_df["sequence"].tolist())

    # Ensure the output order is the same as the input order
    for i, score_dict in enumerate(all_sequences_scored.to_dict(orient="records")):
        assert (
            score_dict["sequence"] == all_sequences[i]
        ), f"Incorrect ordering: {score_dict['sequence']} != {all_sequences[i]}"

    # Ensure that sequences that are the same have the same scores
    unique_sequences = set(all_sequences_df["sequence"].tolist())
    for u_seq in unique_sequences:
        scores = {}
        for seq_dict in all_sequences_scored.to_dict(orient="records"):
            if seq_dict["sequence"] == u_seq:
                if scores.get(seq_dict["sequence"]) is None:
                    scores = seq_dict.copy()
                else:
                    for key, value in seq_dict.items():
                        assert scores[key] == value, f"Expected {scores[key]} but got {value}"

    # Ensure that the cache length is equal to the number of unique sequences
    assert len(evaluator.cache) == len(
        unique_sequences
    ), f"Expected {len(unique_sequences)} but got {len(evaluator.cache)}"


@pytest.mark.parametrize("evaluator_class", EVALUATOR_REGISTRY.list_available_classes())
def test_available_scores(evaluator_class):
    """
    Tests the available_scores method for each evaluator
    """
    available_scores = evaluator_class.available_scores()
    assert isinstance(available_scores, dict)

    # Ensure each score has the correct dict structure
    for score_key, score_info in available_scores.items():
        assert isinstance(score_key, str)
        assert isinstance(score_info, dict)
        assert "larger_is_better" in score_info
        assert "min_value" in score_info
        assert "max_value" in score_info


@pytest.mark.requires_gpu
@pytest.mark.parametrize("evaluator_class", EVALUATOR_REGISTRY.list_available_classes())
def test_evaluators(evaluator_class):

    seed_everything(42)

    evaluator = evaluator_class(seed=42)

    if "AAV" not in evaluator_class.__name__:
        sequences = [GFP] * 10
        mutated_sequences = [mutate_k(sequence, k=10) for sequence in sequences]
    else:
        sequences = [AAV_WT_VP3] * 10
        mutated_sequences = [
            mutate_k(
                sequence,
                k=10,
                fixed_indices=get_fixed_positions_from_file(
                    file="inputs/aav/fixed_positions_aav2.txt",
                    assume_1_indexed=True,
                ),
            )
            for sequence in sequences
        ]

    other_field = [i for i in range(len(mutated_sequences))]

    df_sequences = pd.DataFrame({"sequence": mutated_sequences})
    df_sequences["other_field"] = other_field

    other_kwargs = {}
    if (
        evaluator_class.__name__ == "SequenceSimilarity"
        or evaluator_class.__name__ == "DiscountedHammingDistance"
    ):
        other_kwargs["reference_sequence_or_sequences"] = GFP

    # Run on the first sequences
    input_copy = df_sequences.copy(deep=True)
    scored_sequences_df = evaluator.evaluate(sequences=df_sequences, **other_kwargs)

    # Ensure input is not modified
    assert df_sequences.equals(input_copy), "Input should not be modified"

    assert len(scored_sequences_df) == len(
        mutated_sequences
    ), f"Expected {len(mutated_sequences)} scored sequences but got {len(scored_sequences_df)}"
    # Ensure that the other field is conserved
    assert (
        scored_sequences_df["other_field"].tolist() == other_field
    ), f"Expected {other_field} but got {scored_sequences_df['other_field'].tolist()}"

    # Run on sequences again and ensure that they are the same
    scored_sequences_2_df = evaluator.evaluate(sequences=df_sequences, **other_kwargs)
    assert len(scored_sequences_2_df) == len(
        mutated_sequences
    ), f"Expected {len(mutated_sequences)} scored sequences but got {len(scored_sequences_2_df)}"

    # Ensure that the scores are the same
    assert df_sequences.equals(input_copy), "Input should not be modified"
    assert scored_sequences_df.equals(scored_sequences_2_df), "Outputs should be the same"

    # Run evaluator on new sequences
    if evaluator_class.__name__ == "AAVCapsidViability":
        new_sequences = [
            mutate_k(
                sequence,
                k=10,
                fixed_indices=get_fixed_positions_from_file(
                    file="inputs/aav/fixed_positions_aav2.txt", assume_1_indexed=True
                ),
            )
            for sequence in sequences
        ]
    else:
        new_sequences = [mutate_k(sequence, k=10) for sequence in sequences]

    scored_sequences_3_df = evaluator.evaluate(sequences=new_sequences, **other_kwargs)
    assert len(scored_sequences_3_df) == len(
        new_sequences
    ), f"Expected {len(new_sequences)} scored sequences but got {len(scored_sequences_3_df)}"

    # Ensure that the cache is equal in length to the number of unique sequences
    num_unique_sequences = set(mutated_sequences + new_sequences)

    if evaluator.use_cache:
        assert len(evaluator.cache) == len(
            num_unique_sequences
        ), f"Expected {len(num_unique_sequences)} but got {len(evaluator.cache)}"
    else:
        assert len(evaluator.cache) == len(
            set(new_sequences)
        ), f"Expected {len(set(new_sequences))} but got {len(evaluator.cache)}"

    # Ensure that the other field is not present in the last one (since it was not passed in)
    assert (
        "other_field" not in scored_sequences_3_df.columns
    ), "Other field should not be present since the input was a list"

    # Ensure that all 'active_scores' are present in the last one
    assert set(list(evaluator.active_scores)).issubset(
        set(scored_sequences_3_df.columns)
    ), f"Expected {evaluator.active_scores} to be present in {set(scored_sequences_3_df.columns)}"

    # For each active score, ensure that all scores are valid if min_value or max_value is not None
    for score in evaluator.active_scores:
        if isinstance(evaluator.available_scores()[score]["min_value"], (int, float)):
            assert (
                scored_sequences_3_df[score].min()
                >= evaluator.available_scores()[score]["min_value"]
            ), f"Expected {evaluator.available_scores()[score]['min_value']} but got {scored_sequences_3_df[score].min()}"
        if isinstance(evaluator.available_scores()[score]["max_value"], (int, float)):
            assert (
                scored_sequences_3_df[score].max()
                <= evaluator.available_scores()[score]["max_value"]
            ), f"Expected {evaluator.available_scores()[score]['max_value']} but got {scored_sequences_3_df[score].max()}"

        # If both are specified, ensure that min_value is less than max_value
        if isinstance(
            evaluator.available_scores()[score]["min_value"], (int, float)
        ) and isinstance(evaluator.available_scores()[score]["max_value"], (int, float)):
            assert (
                evaluator.available_scores()[score]["min_value"]
                < evaluator.available_scores()[score]["max_value"]
            ), f"Expected {evaluator.available_scores()[score]['min_value']} to be less than {evaluator.available_scores()[score]['max_value']}"

    # CONSISTENCY CHECK
    # Ensure that the same results are returned for the same sequences

    # Skip the consistency check for the structure metrics evaluator since it is not deterministic
    if evaluator_class.__name__ == "StructureMetrics":
        return

    # First, clear the cache
    evaluator.cache = None

    comparison_df = evaluator.evaluate(sequences=df_sequences, **other_kwargs)

    # Ensure that sequences have the same scores and ordering
    assert scored_sequences_df.equals(
        comparison_df
    ), "Sequences should have the same scores and ordering"


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


def test_sequence_similarity_evaluator_correctness():
    """
    Tests the correctness of the sequence similarity evaluator
    """

    evaluator = EVALUATOR_REGISTRY.get_instance("sequence_similarity")

    for test_case in TEST_CASES:
        df = evaluator.evaluate(
            sequences=[test_case["seq1"]],
            reference_sequence_or_sequences=test_case["seq2"],
        )

        # Check values
        for key, value in test_case["expected"].items():
            assert (
                df[key].iloc[0] == value
            ), f"{key} incorrect: Expected {value} but got {df[key].iloc[0]}"


def test_evaluator_instance_cache():
    """
    Tests the evaluator instance cache
    """
    clear_all_evaluator_cache()
    assert len(EVALUATOR_INSTANCE_CACHE) == 0

    evaluator1 = get_evaluator("sequence_similarity", active_scores=["sequence_similarity"])

    evaluator2 = get_evaluator(
        "sequence_similarity", active_scores=["normalized_hamming_distance"]
    )

    # Ensure the instance cache contains one entry
    assert len(EVALUATOR_INSTANCE_CACHE) == 1

    evaluator3 = get_evaluator("dummy", use_cache=True)
    evaluator4 = get_evaluator("dummy", use_cache=False)

    # Ensure the instance cache contains two entries
    assert len(EVALUATOR_INSTANCE_CACHE) == 2

    # Ensure evaluator1 and evaluator2 are the same instance
    assert evaluator1 is evaluator2

    # Ensure they contain the correct active scores
    assert evaluator1.active_scores == ["sequence_similarity", "normalized_hamming_distance"]

    # Ensure evaluator3 and evaluator4 are the same instance
    assert evaluator3 is evaluator4

    # Ensure evaluator3 and 4 are using their result caches
    assert evaluator3.use_cache

    # Ensure we can clear the cache as well
    clear_all_evaluator_cache()

    # Ensure the instance cache is now empty
    assert len(EVALUATOR_INSTANCE_CACHE) == 0

    # Ensure the cache is empty
    assert len(evaluator1.cache) == 0
    assert len(evaluator3.cache) == 0


def test_sequence_similarity_only_consider_designable():
    """
    Tests that only_consider_designable correctly filters sequences to designable positions
    """

    test_sequence = "ABCDEFGH"
    reference_sequence = "ABCDWXYZ"
    fixed_position_indices = [0, 1, 2, 3]
    sequence_length = 8

    # Create evaluator with only_consider_designable=True
    evaluator_designable_only = EVALUATOR_REGISTRY.get_instance(
        "sequence_similarity",
        only_consider_designable=True,
        active_scores=["sequence_identity", "normalized_hamming_distance"],
    )

    # Evaluate with only_consider_designable=True
    df_designable = evaluator_designable_only.evaluate(
        sequences=[test_sequence],
        reference_sequence_or_sequences=reference_sequence,
        fixed_position_indices=fixed_position_indices,
        sequence_length=sequence_length,
    )

    # When considering only designable positions:
    # Test:      "EFGH"
    # Reference: "WXYZ"
    # All 4 positions differ, so identity should be 0.0 and normalized hamming should be 1.0
    assert (
        df_designable["sequence_identity"].iloc[0] == 0.0
    ), f"Expected sequence_identity=0.0 but got {df_designable['sequence_identity'].iloc[0]}"
    assert (
        df_designable["normalized_hamming_distance"].iloc[0] == 1.0
    ), f"Expected normalized_hamming_distance=1.0 but got {df_designable['normalized_hamming_distance'].iloc[0]}"

    # Create evaluator with only_consider_designable=False
    evaluator_all_positions = EVALUATOR_REGISTRY.get_instance(
        "sequence_similarity",
        only_consider_designable=False,
        active_scores=["sequence_identity", "normalized_hamming_distance"],
    )

    # Evaluate with only_consider_designable=False
    df_all = evaluator_all_positions.evaluate(
        sequences=[test_sequence],
        reference_sequence_or_sequences=reference_sequence,
        fixed_position_indices=fixed_position_indices,
        sequence_length=sequence_length,
    )

    # When considering all positions:
    # Test:      "ABCDEFGH"
    # Reference: "ABCDWXYZ"
    # 4 positions match (ABCD), 4 differ (EFGH vs WXYZ)
    # Identity should be 0.5 and normalized hamming should be 0.5
    assert (
        df_all["sequence_identity"].iloc[0] == 0.5
    ), f"Expected sequence_identity=0.5 but got {df_all['sequence_identity'].iloc[0]}"
    assert (
        df_all["normalized_hamming_distance"].iloc[0] == 0.5
    ), f"Expected normalized_hamming_distance=0.5 but got {df_all['normalized_hamming_distance'].iloc[0]}"


def test_sequence_similarity_with_list_of_references():
    """
    Tests that SequenceSimilarity can handle a list of reference sequences
    """
    evaluator = EVALUATOR_REGISTRY.get_instance(
        "sequence_similarity",
        active_scores=["sequence_identity", "normalized_hamming_distance"],
    )

    # Test with multiple sequences and matching number of reference sequences
    # Using valid amino acids only: ACDEFGHIKLMNPQRSTVWY
    test_sequences = ["ACDEFGHI", "IKLMNPQR", "QRSTVWY"]
    reference_sequences = ["ACDEWYHI", "IKLMACDE", "QRSTVWY"]

    df = evaluator.evaluate(
        sequences=test_sequences,
        reference_sequence_or_sequences=reference_sequences,
    )

    # First sequence: "ACDEFGHI" vs "ACDEWYHI" -> 6/8 = 0.75 identity
    assert (
        df["sequence_identity"].iloc[0] == 0.75
    ), f"Expected sequence_identity=0.75 for first sequence but got {df['sequence_identity'].iloc[0]}"

    # Second sequence: "IKLMNPQR" vs "IKLMACDE" -> 4/8 = 0.5 identity
    assert (
        df["sequence_identity"].iloc[1] == 0.5
    ), f"Expected sequence_identity=0.5 for second sequence but got {df['sequence_identity'].iloc[1]}"

    # Third sequence: "QRSTVWY" vs "QRSTVWY" -> 7/7 = 1.0 identity
    assert (
        df["sequence_identity"].iloc[2] == 1.0
    ), f"Expected sequence_identity=1.0 for third sequence but got {df['sequence_identity'].iloc[2]}"


def test_sequence_similarity_designable_with_list_of_references():
    """
    Tests that only_consider_designable works correctly with a list of reference sequences
    """
    evaluator = EVALUATOR_REGISTRY.get_instance(
        "sequence_similarity",
        only_consider_designable=True,
        active_scores=["sequence_identity"],
    )

    # Test sequences where fixed positions match but designable positions differ
    # Using valid amino acids only: ACDEFGHIKLMNPQRSTVWY
    test_sequences = ["ACDEFGHI", "ACDEIKLM"]
    reference_sequences = ["ACDEWYHI", "ACDEMNPQ"]
    fixed_position_indices = [0, 1, 2, 3]
    sequence_length = 8

    df = evaluator.evaluate(
        sequences=test_sequences,
        reference_sequence_or_sequences=reference_sequences,
        fixed_position_indices=fixed_position_indices,
        sequence_length=sequence_length,
    )

    # Both should have 0.0 identity since all designable positions differ
    # First: "FGHI" vs "WYHI" -> "I" matches at position 3, so 1/4 = 0.25 identity
    # Wait, let me recalculate:
    # test_sequences[0] = "ACDEFGHI", designable = "FGHI"
    # reference_sequences[0] = "ACDEWYHI", designable = "WYHI"
    # "FGHI" vs "WYHI" -> only "HI" matches at positions 2,3 -> 2/4 = 0.5 identity

    # Let me adjust the sequences to make them completely different in designable positions
    # Actually, let me recalculate with the current sequences
    assert (
        df["sequence_identity"].iloc[0] == 0.5
    ), f"Expected sequence_identity=0.5 for first sequence but got {df['sequence_identity'].iloc[0]}"

    # Second: "IKLM" vs "MNPQ" -> all different -> 0.0 identity
    assert (
        df["sequence_identity"].iloc[1] == 0.0
    ), f"Expected sequence_identity=0.0 for second sequence but got {df['sequence_identity'].iloc[1]}"


def test_sequence_similarity_without_fixed_positions():
    """
    Tests that only_consider_designable=True with no fixed positions uses full sequences
    """
    evaluator = EVALUATOR_REGISTRY.get_instance(
        "sequence_similarity",
        only_consider_designable=True,
        active_scores=["sequence_identity"],
    )

    test_sequence = "ABCDEFGH"
    reference_sequence = "ABCDWXYZ"

    # Don't provide fixed_position_indices - should use full sequences
    df = evaluator.evaluate(
        sequences=[test_sequence],
        reference_sequence_or_sequences=reference_sequence,
    )

    # Should compare full sequences: "ABCDEFGH" vs "ABCDWXYZ" -> 4/8 = 0.5
    assert (
        df["sequence_identity"].iloc[0] == 0.5
    ), f"Expected sequence_identity=0.5 but got {df['sequence_identity'].iloc[0]}"


def test_sequence_similarity_partial_designable_positions():
    """
    Tests case where only some positions are designable
    """
    evaluator = EVALUATOR_REGISTRY.get_instance(
        "sequence_similarity",
        only_consider_designable=True,
        active_scores=[
            "sequence_identity",
            "levenshtein_distance",
            "normalized_hamming_distance",
        ],
    )

    # Using valid amino acids only: ACDEFGHIKLMNPQRSTVWY
    test_sequence = "ACDEFGHI"
    reference_sequence = "ACDEWYHI"
    fixed_position_indices = [0, 1, 2, 3]  # First 4 positions fixed
    sequence_length = 8

    df = evaluator.evaluate(
        sequences=[test_sequence],
        reference_sequence_or_sequences=reference_sequence,
        fixed_position_indices=fixed_position_indices,
        sequence_length=sequence_length,
    )

    # Designable positions [4, 5, 6, 7]: "FGHI" vs "WYHI"
    # Only "HI" match at positions 2-3 of the designable region -> 2/4 = 0.5 identity
    assert (
        df["sequence_identity"].iloc[0] == 0.5
    ), f"Expected sequence_identity=0.5 but got {df['sequence_identity'].iloc[0]}"

    # Normalized hamming distance should be 0.5 (2 out of 4 positions differ)
    assert (
        df["normalized_hamming_distance"].iloc[0] == 0.5
    ), f"Expected normalized_hamming_distance=0.5 but got {df['normalized_hamming_distance'].iloc[0]}"

    # Levenshtein distance should be 2 (2 substitutions needed)
    assert (
        df["levenshtein_distance"].iloc[0] == 2
    ), f"Expected levenshtein_distance=2 but got {df['levenshtein_distance'].iloc[0]}"
