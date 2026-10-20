"""Sample sequences with ProVADA's JAX/ColabDesign ProteinMPNN."""

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


class ProteinMPNNJaxSampleConfig(ProteinMPNNSampleConfigBase):
    """Sampling config for the JAX toolkit."""

#: Structure used for the registry's example input. It lives in ``inputs/``,
#: outside the package, so this resolves only in a source checkout.
_EXAMPLE_STRUCTURE = REPO_ROOT / "inputs" / "renin" / "renin_af3.pdb"


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNSampleInput(structure=str(_EXAMPLE_STRUCTURE))


@tool(
    key="proteinmpnn-jax-sample",
    label="ProteinMPNN (JAX) Sampling",
    category="inverse_folding",
    input_class=ProteinMPNNSampleInput,
    config_class=ProteinMPNNJaxSampleConfig,
    output_class=ProteinMPNNSampleOutput,
    description="Sample sequences with the JAX/ColabDesign ProteinMPNN",
    uses_gpu=True,
    pin_visible_devices=True,
    stochastic=True,
    example_input=example_input,
)
def run_proteinmpnn_jax_sample(
    inputs: ProteinMPNNSampleInput,
    config: ProteinMPNNJaxSampleConfig,
    instance: Any = None,
) -> ProteinMPNNSampleOutput:
    """Design sequences for a structure, pinning the prompt at fixed positions.

    ``pin_visible_devices`` is set because a JAX runtime initializes a context
    on every visible GPU, so this worker is given one device and addresses it as
    ``cuda:0``.
    """
    result = ToolInstance.dispatch(
        "proteinmpnn_jax",
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
