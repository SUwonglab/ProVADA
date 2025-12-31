"""
test_sampler.py

Tests the Sampler class
"""

import pytest
from provada.sampler.sampler import SAMPLER_REGISTRY
from provada.base_variant import BaseVariant
from provada.utils.setup import init_wandb
from provada.utils.env import suppress_console_output

SCORE_WEIGHTS = [
    {"dummy_score": 1.0, "sequence_similarity": 0.0},
    {"dummy_score": 1.0, "sequence_similarity": 0.2},
    {"dummy_score": 1.0, "sequence_similarity": 0.5},
    {"dummy_score": 1.0, "sequence_similarity": 10.0},
]

test_pdb_path = "inputs/renin/renin_af3.pdb"


@pytest.mark.parametrize(
    "sampler_name,score_weights",
    [
        (sampler_name, score_weights)
        for sampler_name in SAMPLER_REGISTRY.list_available_class_names()
        for score_weights in SCORE_WEIGHTS
    ],
)
def test_sampler(sampler_name, score_weights):

    test_config = {
        "num_iters": 100,
        "population_size": 100,
        "top_k_fraction": 0.2,
        "score_weights": score_weights,
        "generator_type": "random",
        "generation_kwargs": {"codon_scheme": "NNK"},
        "temperature_schedule_config": {
            "type": "power",
            "kwargs": {"alpha": 3.0, "start_value": 5.0, "stop_value": 0.1},
        },
        "seed": 42,
        "device": "cpu",
        "run_name": f"test_{sampler_name}_seqsim_{score_weights['sequence_similarity']}",
        "output_dir": "results/test_results",
        "sampler_name": sampler_name,
        "masking_strategy_type": "gaussian_thompson",
        "masking_strategy_kwargs": {"alpha": 2.0, "gamma": 0.95},
        "masking_schedule_config": {
            "type": "power",
            "kwargs": {"start_value": 0.2, "stop_value": 0.05, "alpha": 2.0},
        },
    }

    # Initialize wandb run
    wandb_run = init_wandb(
        no_wandb=False,
        run_name=f"test_{sampler_name}_seqsim_{score_weights['sequence_similarity']}",
        config=test_config,
        wandb_project="provada",
    )

    # Initialize sampler
    SamplerClass = SAMPLER_REGISTRY.get_class(sampler_name)
    sampler = SamplerClass(
        base_variant=BaseVariant(
            sequence="DARTHVADERPETERPARKER", structure=test_pdb_path, fixed_indices=[0, 5]
        ),
        num_iters=test_config["num_iters"],
        population_size=test_config["population_size"],
        top_k_fraction=test_config["top_k_fraction"],
        score_weights=test_config["score_weights"],
        generator_type=test_config["generator_type"],
        generation_kwargs=test_config["generation_kwargs"],
        masking_strategy_type=test_config["masking_strategy_type"],
        masking_strategy_kwargs=test_config["masking_strategy_kwargs"],
        masking_schedule_config=test_config["masking_schedule_config"],
        temperature_schedule_config=test_config["temperature_schedule_config"],
        seed=test_config["seed"],
        device=test_config["device"],
        run_name=test_config["run_name"],
        output_dir=test_config["output_dir"],
        wandb_run=wandb_run,
    )

    # Run the sampler
    with suppress_console_output():
        sampler.run()
