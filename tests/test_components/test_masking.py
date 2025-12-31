"""
test_masker.py

Tests for masking classes in provada.sampler.masker
"""

import pytest

from provada.components.masking import MASK_STRATEGY_REGISTRY

MASK_STRATEGY_REGISTRY.list_available_class_names()


@pytest.mark.parametrize(
    "masking_strategy_name", MASK_STRATEGY_REGISTRY.list_available_class_names()
)
def test_masking_strategy_init_with_fixed_positions(masking_strategy_name):
    masking_strategy_class = MASK_STRATEGY_REGISTRY.get_class(masking_strategy_name)
    masking_strategy = masking_strategy_class(
        sequence_length=10, fixed_position_indices=[0, 5, 9]
    )
    assert masking_strategy.fixed_position_indices == [0, 5, 9]
    assert masking_strategy.designable_positions == [1, 2, 3, 4, 6, 7, 8]


@pytest.mark.parametrize(
    "masking_strategy_name", MASK_STRATEGY_REGISTRY.list_available_class_names()
)
def test_masking_strategy_update(masking_strategy_name):
    masking_strategy_class = MASK_STRATEGY_REGISTRY.get_class(masking_strategy_name)
    masking_strategy = masking_strategy_class(sequence_length=6, mask_char="_")
    masked_strings = ["_ARGE_", "TAR_ET"]
    rewards = [1.0, 2.0]

    masking_strategy.update(masked_strings, rewards)

    assert masking_strategy.number_of_updates == 1
    assert masking_strategy.number_of_samples == 2


@pytest.mark.parametrize(
    "masking_strategy_name", MASK_STRATEGY_REGISTRY.list_available_class_names()
)
def test_masking_strategy_get_position_stats(masking_strategy_name):
    masking_strategy_class = MASK_STRATEGY_REGISTRY.get_class(masking_strategy_name)
    masking_strategy = masking_strategy_class(sequence_length=5)
    df = masking_strategy.get_position_stats()
    assert len(df.columns) == 5
    assert "selection_count" in df.index


@pytest.mark.parametrize(
    "masking_strategy_name", MASK_STRATEGY_REGISTRY.list_available_class_names()
)
def test_masking_strategy_create_masked_sequences(masking_strategy_name):
    masking_strategy_class = MASK_STRATEGY_REGISTRY.get_class(masking_strategy_name)
    masking_strategy = masking_strategy_class(sequence_length=6)
    sequences = ["TARGET", "SAMPLE"]
    num_masked_sites = 2

    masked_sequences = masking_strategy.create_masked_sequences(sequences, num_masked_sites)

    assert len(masked_sequences) == 2
    for seq in masked_sequences["prompt"].tolist():
        assert len(seq) == 6
        assert seq.count("_") == num_masked_sites


@pytest.mark.parametrize(
    "masking_strategy_name", MASK_STRATEGY_REGISTRY.list_available_class_names()
)
def test_masking_strategy_with_fixed_positions(masking_strategy_name):
    masking_strategy_class = MASK_STRATEGY_REGISTRY.get_class(masking_strategy_name)
    masking_strategy = masking_strategy_class(sequence_length=6, fixed_position_indices=[0, 5])
    sequences = ["TARGET"]
    num_masked_sites = 2

    output_df = masking_strategy.create_masked_sequences(sequences, num_masked_sites)

    masked_sequences = output_df["prompt"].tolist()

    # Fixed positions should not be masked
    assert masked_sequences[0][0] == "T", "Fixed position should not be masked"
    assert masked_sequences[0][5] == "T", "Fixed position should not be masked"
    assert (
        masked_sequences[0].count("_") == num_masked_sites
    ), "Number of masked sites should be equal to the number of masked sites"


@pytest.mark.parametrize("num_masked_sites", [0, 1, 3, 5])
@pytest.mark.parametrize(
    "masking_strategy_name", MASK_STRATEGY_REGISTRY.list_available_class_names()
)
def test_masking_strategy_various_mask_counts(masking_strategy_name, num_masked_sites):
    masking_strategy_class = MASK_STRATEGY_REGISTRY.get_class(masking_strategy_name)
    masking_strategy = masking_strategy_class(sequence_length=6)
    sequences = ["TARGET"]

    output_df = masking_strategy.create_masked_sequences(sequences, num_masked_sites)

    masked_sequences = output_df["prompt"].tolist()

    assert (
        masked_sequences[0].count("_") == num_masked_sites
    ), "Number of masked sites should be equal to the number of masked sites"
