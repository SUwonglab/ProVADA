"""Sample sequences with ProVADA's PyTorch ProteinMPNN fork."""

from typing import Any

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance

from provada.paths import REPO_ROOT

from provada.tools.inverse_folding.shared_data_models import (
    ProteinMPNNSampleConfigBase,
    ProteinMPNNSampleInput,
    ProteinMPNNSampleOutput,
    build_dispatch_input,
)

#: Structure used for the registry's example input. It lives in ``inputs/``,
#: outside the package, so this resolves only in a source checkout.
_EXAMPLE_STRUCTURE = REPO_ROOT / "inputs" / "renin" / "renin_af3.pdb"


class ProteinMPNNPytorchSampleConfig(ProteinMPNNSampleConfigBase):
    """Sampling config for the PyTorch toolkit.

    ``weights="antibody"`` is not available here: AbMPNN is loaded through
    ColabDesign, so it lives in the JAX toolkit.
    """


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNSampleInput(structure=str(_EXAMPLE_STRUCTURE))


@tool(
    key="proteinmpnn-pytorch-sample",
    label="ProteinMPNN (PyTorch) Sampling",
    category="inverse_folding",
    input_class=ProteinMPNNSampleInput,
    config_class=ProteinMPNNPytorchSampleConfig,
    output_class=ProteinMPNNSampleOutput,
    description="Sample sequences with ProVADA's PyTorch ProteinMPNN fork",
    uses_gpu=True,
    stochastic=True,
    example_input=example_input,
)
def run_proteinmpnn_pytorch_sample(
    inputs: ProteinMPNNSampleInput,
    config: ProteinMPNNPytorchSampleConfig,
    instance: Any = None,
) -> ProteinMPNNSampleOutput:
    """Design sequences for a structure, conditioning on the whole prompt.

    Every non-masked residue in the prompt overrides the sequence parsed from
    the structure, which is what lets the model condition on the current
    variant rather than the wild type in the PDB.
    """
    result = ToolInstance.dispatch(
        "proteinmpnn_pytorch",
        build_dispatch_input(inputs, config, "sample"),
        instance=instance,
        config=config,
    )
    return ProteinMPNNSampleOutput(
        sequences=result["sequences"],
        scores=result["scores"],
        metadata={
            "weights": config.weights,
            "temperature": config.temperature,
            "prompt_supplied": inputs.prompt is not None,
        },
    )
