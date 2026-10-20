"""Score sequences against a structure with ProVADA's JAX ProteinMPNN."""

from typing import Any

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance

from provada.tools.inverse_folding.proteinmpnn_jax.proteinmpnn_jax_sample import (
    _EXAMPLE_STRUCTURE,
)
from provada.tools.inverse_folding.shared_data_models import (
    ProteinMPNNScoreConfigBase,
    ProteinMPNNScoreInput,
    ProteinMPNNScoreOutput,
    build_dispatch_input,
)


class ProteinMPNNJaxScoreConfig(ProteinMPNNScoreConfigBase):
    """Scoring config for the JAX toolkit."""


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNScoreInput(
        structure=str(_EXAMPLE_STRUCTURE),
        sequences=["LTLGNTTSSVILTNYMDTQYYGEIGIGTPPQTFKVVFDTGSSNVWVPSSKC"],
    )


@tool(
    key="proteinmpnn-jax-score",
    label="ProteinMPNN (JAX) Scoring",
    category="inverse_folding",
    input_class=ProteinMPNNScoreInput,
    config_class=ProteinMPNNJaxScoreConfig,
    output_class=ProteinMPNNScoreOutput,
    description="Score sequences against a structure with the JAX ProteinMPNN",
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_input,
)
def run_proteinmpnn_jax_score(
    inputs: ProteinMPNNScoreInput,
    config: ProteinMPNNJaxScoreConfig,
    instance: Any = None,
) -> ProteinMPNNScoreOutput:
    """Return one negative average log-likelihood per input sequence."""
    result = ToolInstance.dispatch(
        "proteinmpnn_jax",
        build_dispatch_input(inputs, config, "score"),
        instance=instance,
        config=config,
    )
    return ProteinMPNNScoreOutput(
        scores=result["scores"],
        metadata={"weights": config.weights},
    )
