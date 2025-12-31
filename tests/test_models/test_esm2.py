"""
test_esm3.py

Tests the ESM3 implementation
"""

from provada.models.esm2 import ESM2Model
import pytest


@pytest.mark.requires_gpu
def test_esm2_forward_pass():

    sequences = ["TARGET"] * 10 + ["TEST"] * 30

    esm2_model = ESM2Model(model_name="esm2_t33_650M_UR50D")

    outputs = esm2_model(sequences, batch_size=2, return_logits=True)

    # Check mean embedding shape
    assert outputs["mean_embeddings"].shape == (
        40,
        1280,
    ), "Embedding shape is not correct"

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
