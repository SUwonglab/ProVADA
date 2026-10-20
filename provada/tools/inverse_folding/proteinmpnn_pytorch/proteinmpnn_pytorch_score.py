"""Score sequences against a structure with ProVADA's PyTorch ProteinMPNN."""

from typing import Any

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance

from provada.tools.inverse_folding.proteinmpnn_pytorch.proteinmpnn_pytorch_sample import (
    _EXAMPLE_STRUCTURE,
)
from provada.tools.inverse_folding.shared_data_models import (
    ProteinMPNNScoreConfigBase,
    ProteinMPNNScoreInput,
    ProteinMPNNScoreOutput,
    build_dispatch_input,
)


class ProteinMPNNPytorchScoreConfig(ProteinMPNNScoreConfigBase):
    """Scoring config for the PyTorch toolkit."""


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNScoreInput(
        structure=str(_EXAMPLE_STRUCTURE),
        sequences=["LTLGNTTSSVILTNYMDTQYYGEIGIGTPPQTFKVVFDTGSSNVWVPSSKC"],
    )


@tool(
    key="proteinmpnn-pytorch-score",
    label="ProteinMPNN (PyTorch) Scoring",
    category="inverse_folding",
    input_class=ProteinMPNNScoreInput,
    config_class=ProteinMPNNPytorchScoreConfig,
    output_class=ProteinMPNNScoreOutput,
    description="Score sequences against a structure with the PyTorch ProteinMPNN",
    uses_gpu=True,
    example_input=example_input,
)
def run_proteinmpnn_pytorch_score(
    inputs: ProteinMPNNScoreInput,
    config: ProteinMPNNPytorchScoreConfig,
    instance: Any = None,
) -> ProteinMPNNScoreOutput:
    """Return one negative average log-likelihood per input sequence."""
    result = ToolInstance.dispatch(
        "proteinmpnn_pytorch",
        build_dispatch_input(inputs, config, "score"),
        instance=instance,
        config=config,
    )
    return ProteinMPNNScoreOutput(
        scores=result["scores"],
        metadata={"weights": config.weights},
    )
