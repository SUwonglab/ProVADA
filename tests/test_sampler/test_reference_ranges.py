"""
test_reference_ranges.py

Tests the reference ranges optimization in the ReferenceRangesMixin class.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from provada.sampler.sampler import SAMPLER_REGISTRY
from provada.base_variant import BaseVariant
from provada.utils.setup import init_wandb


test_pdb_path = "inputs/renin/renin_af3.pdb"


def test_reference_ranges_only_evaluates_needed_scores():
    """
    Test that when determining reference ranges, only evaluators for scores
    with unspecified ranges are run, not all evaluators.

    This test verifies the optimization that avoids running expensive evaluators
    when their scores already have specified min/max ranges.
    """

    test_config = {
        "num_iters": 2,
        "population_size": 10,
        "top_k_fraction": 0.5,
        "score_weights": {
            "dummy_score": 1.0,
            "sequence_similarity": 0.5,
        },
        "generator_type": "random",
        "generation_kwargs": {"codon_scheme": "NNK"},
        "temperature_schedule_config": {
            "type": "power",
            "kwargs": {"alpha": 3.0, "start_value": 5.0, "stop_value": 0.1},
        },
        "seed": 42,
        "device": "cpu",
        "run_name": "test_reference_ranges_optimization",
        "output_dir": tempfile.mkdtemp(),
        "sampler_name": "rejection",
        "masking_strategy_type": "gaussian_thompson",
        "masking_strategy_kwargs": {"alpha": 2.0, "gamma": 0.95},
        "masking_schedule_config": {
            "type": "power",
            "kwargs": {"start_value": 0.2, "stop_value": 0.05, "alpha": 2.0},
        },
    }

    try:
        # Initialize wandb run
        wandb_run = init_wandb(
            no_wandb=True,
            run_name=test_config["run_name"],
            config=test_config,
            wandb_project="provada-test",
        )

        # Initialize sampler WITHOUT calling determine_min_max_ranges yet
        SamplerClass = SAMPLER_REGISTRY.get_class(test_config["sampler_name"])

        # We'll create the sampler but intercept the initialization to modify ranges
        # before they're finalized
        sampler = SamplerClass.__new__(SamplerClass)

        # Manually initialize only the parts we need for testing
        sampler.base_variant = BaseVariant(
            sequence="DARTHVADERPETERPARKER",
            structure=test_pdb_path,
            fixed_indices=[0, 5],
        )
        sampler.score_weights = test_config["score_weights"]
        sampler.seed = test_config["seed"]
        sampler.output_dir = Path(test_config["output_dir"])
        sampler.wandb_run = wandb_run
        sampler.reference_distribution_file = None

        # Initialize active_scores_dict
        sampler.active_scores_dict = {
            "dummy_score": {"weight": 1.0, "min_value": None, "max_value": None},
            "sequence_similarity": {"weight": 0.5, "min_value": 0.0, "max_value": 1.0},
        }

        # Initialize evaluators and generator (needed for the test)
        from provada.sampler.startup import StartupMixin

        sampler.evaluators = []
        StartupMixin.initialize_evaluators(sampler)

        # Initialize generator (needed for reference distribution generation)
        StartupMixin.initialize_generator(sampler, "random", {"codon_scheme": "NNK"})

        # Now manually set the ranges we want to test
        # dummy_score: ranges unspecified (None) - should need evaluation
        # sequence_similarity: ranges specified - should NOT need evaluation
        sampler.active_scores_dict["dummy_score"]["min_value"] = None
        sampler.active_scores_dict["dummy_score"]["max_value"] = None
        sampler.active_scores_dict["sequence_similarity"]["min_value"] = 0.0
        sampler.active_scores_dict["sequence_similarity"]["max_value"] = 1.0

        # Call determine_min_max_ranges
        from provada.sampler.reference_ranges import ReferenceRangesMixin

        ReferenceRangesMixin._determine_min_max_ranges(sampler)

        # Verify that min/max ranges were set for dummy_score
        assert sampler.active_scores_dict["dummy_score"]["min_value"] is not None
        assert sampler.active_scores_dict["dummy_score"]["max_value"] is not None
        assert isinstance(sampler.active_scores_dict["dummy_score"]["min_value"], (int, float))
        assert isinstance(sampler.active_scores_dict["dummy_score"]["max_value"], (int, float))

        # Verify that sequence_similarity kept its specified ranges
        assert sampler.active_scores_dict["sequence_similarity"]["min_value"] == 0.0
        assert sampler.active_scores_dict["sequence_similarity"]["max_value"] == 1.0

        # Verify the reference distribution file was created
        ref_dist_path = Path(test_config["output_dir"]) / "reference_distribution.csv"
        assert ref_dist_path.exists(), "Reference distribution CSV should be created"

        # Load and verify the reference distribution
        ref_dist = pd.read_csv(ref_dist_path)

        # Should have sequence column and dummy_score
        assert "sequence" in ref_dist.columns, "Reference distribution should have sequences"
        assert (
            "dummy_score" in ref_dist.columns
        ), "Reference distribution should have dummy_score (unspecified range)"

    finally:
        # Clean up temporary directory
        shutil.rmtree(test_config["output_dir"], ignore_errors=True)


def test_no_reference_distribution_when_all_ranges_specified():
    """
    Test that when all score ranges are specified, no reference distribution
    is generated at all.

    This is a key optimization - if the user specifies all min/max ranges,
    we should skip the expensive reference distribution generation entirely.
    """

    test_config = {
        "num_iters": 2,
        "population_size": 10,
        "top_k_fraction": 0.5,
        "score_weights": {
            "dummy_score": 1.0,
            "sequence_similarity": 0.5,
        },
        "generator_type": "random",
        "generation_kwargs": {"codon_scheme": "NNK"},
        "temperature_schedule_config": {
            "type": "power",
            "kwargs": {"alpha": 3.0, "start_value": 5.0, "stop_value": 0.1},
        },
        "seed": 42,
        "device": "cpu",
        "run_name": "test_no_reference_distribution",
        "output_dir": tempfile.mkdtemp(),
        "sampler_name": "rejection",
        "masking_strategy_type": "gaussian_thompson",
        "masking_strategy_kwargs": {"alpha": 2.0, "gamma": 0.95},
        "masking_schedule_config": {
            "type": "power",
            "kwargs": {"start_value": 0.2, "stop_value": 0.05, "alpha": 2.0},
        },
    }

    try:
        # Initialize wandb run
        wandb_run = init_wandb(
            no_wandb=True,
            run_name=test_config["run_name"],
            config=test_config,
            wandb_project="provada-test",
        )

        # Initialize sampler WITHOUT calling determine_min_max_ranges yet
        SamplerClass = SAMPLER_REGISTRY.get_class(test_config["sampler_name"])
        sampler = SamplerClass.__new__(SamplerClass)

        # Manually initialize only the parts we need for testing
        sampler.base_variant = BaseVariant(
            sequence="DARTHVADERPETERPARKER",
            structure=test_pdb_path,
            fixed_indices=[0, 5],
        )
        sampler.score_weights = test_config["score_weights"]
        sampler.seed = test_config["seed"]
        sampler.output_dir = Path(test_config["output_dir"])
        sampler.wandb_run = wandb_run
        sampler.reference_distribution_file = None

        # Initialize active_scores_dict
        sampler.active_scores_dict = {
            "dummy_score": {"weight": 1.0},
            "sequence_similarity": {"weight": 0.5},
        }

        # Initialize evaluators and generator (needed for the test)
        from provada.sampler.startup import StartupMixin

        sampler.evaluators = []
        StartupMixin.initialize_evaluators(sampler)

        # Initialize generator (needed for reference distribution generation if it were to run)
        StartupMixin.initialize_generator(sampler, "random", {"codon_scheme": "NNK"})

        # NOW set ALL ranges to specified values (after evaluator initialization)
        sampler.active_scores_dict["dummy_score"]["min_value"] = -10.0
        sampler.active_scores_dict["dummy_score"]["max_value"] = 10.0
        sampler.active_scores_dict["sequence_similarity"]["min_value"] = 0.0
        sampler.active_scores_dict["sequence_similarity"]["max_value"] = 1.0

        # Call determine_min_max_ranges
        from provada.sampler.reference_ranges import ReferenceRangesMixin

        ReferenceRangesMixin._determine_min_max_ranges(sampler)

        # Verify that the ranges are as specified
        assert sampler.active_scores_dict["dummy_score"]["min_value"] == -10.0
        assert sampler.active_scores_dict["dummy_score"]["max_value"] == 10.0
        assert sampler.active_scores_dict["sequence_similarity"]["min_value"] == 0.0
        assert sampler.active_scores_dict["sequence_similarity"]["max_value"] == 1.0

        # Verify no reference distribution file was created
        ref_dist_path = Path(test_config["output_dir"]) / "reference_distribution.csv"
        assert (
            not ref_dist_path.exists()
        ), "Reference distribution CSV should NOT be created when all ranges specified"

    finally:
        # Clean up temporary directory
        shutil.rmtree(test_config["output_dir"], ignore_errors=True)


def test_evaluate_variants_subset():
    """
    Test that _evaluate_variants_subset only runs the necessary evaluators.

    This is a unit test for the core optimization method.
    """

    test_config = {
        "num_iters": 2,
        "population_size": 10,
        "top_k_fraction": 0.5,
        "score_weights": {
            "dummy_score": 1.0,
            "sequence_similarity": 0.5,
        },
        "generator_type": "random",
        "generation_kwargs": {"codon_scheme": "NNK"},
        "temperature_schedule_config": {
            "type": "power",
            "kwargs": {"alpha": 3.0, "start_value": 5.0, "stop_value": 0.1},
        },
        "seed": 42,
        "device": "cpu",
        "run_name": "test_evaluate_variants_subset",
        "output_dir": tempfile.mkdtemp(),
        "sampler_name": "rejection",
        "masking_strategy_type": "gaussian_thompson",
        "masking_strategy_kwargs": {"alpha": 2.0, "gamma": 0.95},
        "masking_schedule_config": {
            "type": "power",
            "kwargs": {"start_value": 0.2, "stop_value": 0.05, "alpha": 2.0},
        },
    }

    try:
        # Initialize wandb run
        wandb_run = init_wandb(
            no_wandb=True,
            run_name=test_config["run_name"],
            config=test_config,
            wandb_project="provada-test",
        )

        # Initialize sampler properly
        SamplerClass = SAMPLER_REGISTRY.get_class(test_config["sampler_name"])
        sampler = SamplerClass(
            base_variant=BaseVariant(
                sequence="DARTHVADERPETERPARKER",
                structure=test_pdb_path,
                fixed_indices=[0, 5],
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

        # Create test data
        test_df = pd.DataFrame(
            {
                "sequence": ["DARTHVADERPETERPARKER", "AAAAAVADERPETERPARKER"],
            }
        )

        # Test evaluating only dummy_score
        result_df = sampler._evaluate_variants_subset(
            iteration_df=test_df, score_subset={"dummy_score"}
        )

        # Should have sequence and dummy_score
        assert "sequence" in result_df.columns
        assert "dummy_score" in result_df.columns
        assert len(result_df) == 2

        # Test evaluating only sequence_similarity
        result_df = sampler._evaluate_variants_subset(
            iteration_df=test_df, score_subset={"sequence_similarity"}
        )

        # Should have sequence and sequence_similarity
        assert "sequence" in result_df.columns
        assert "sequence_similarity" in result_df.columns
        assert len(result_df) == 2

        # Test evaluating both
        result_df = sampler._evaluate_variants_subset(
            iteration_df=test_df, score_subset={"dummy_score", "sequence_similarity"}
        )

        # Should have sequence and both scores
        assert "sequence" in result_df.columns
        assert "dummy_score" in result_df.columns
        assert "sequence_similarity" in result_df.columns
        assert len(result_df) == 2

    finally:
        # Clean up temporary directory
        shutil.rmtree(test_config["output_dir"], ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
