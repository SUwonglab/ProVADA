"""
test_esm3.py

Tests the ESM3 implementation
"""

import pytest
from provada.models.esm3 import ESM3Model
from provada.sequences.vocab import GFP


@pytest.mark.requires_gpu
def test_esm3_forward_pass():

    sequences = ["TARGET"] * 10 + ["TEST"] * 30

    esm3_model = ESM3Model()

    outputs = esm3_model(sequences, batch_size=2, return_logits=True)

    # Check mean embedding shape
    assert outputs["mean_embeddings"].shape == (
        40,
        1536,
    ), "Avg embedding shape is not correct"

    # Check attention mask shape
    assert outputs["attention_masks"].shape == (
        40,
        6,
    ), "Attention mask shape is not correct"

    # Check logit shape
    assert outputs["logits"].shape == (40, 6, 20), "Logit shape is not correct"

    # Ensure output only contains these keys
    assert set(outputs.keys()) == {
        "mean_embeddings",
        "attention_masks",
        "logits",
    }


@pytest.mark.requires_gpu
def test_esm3_predict_structure():

    sequences = [GFP] * 2

    esm3_model = ESM3Model()

    esm3_model.predict_structure(sequences, batch_size=2)
