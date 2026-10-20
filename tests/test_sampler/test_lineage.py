"""Real CSV round trips for rejected children and retained particle counts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analyze.loading import load_run_data, merge_scores_into_trajectory
from provada.sampler.sampler import MADA
from provada.sequences.io import hash_sequence
from provada.utils.cache import shutdown_cache


@pytest.fixture(autouse=True)
def flush_lineage_cache():
    yield
    shutdown_cache()


def lineage_sampler(tmp_path):
    sampler = MADA.__new__(MADA)
    sampler.resampling_method = "systematic"
    sampler.normalize, sampler.greedy = True, False
    sampler.active_scores_dict = {"reward": dict(weight=1., min_value=0., max_value=1., larger_is_better=True)}
    sampler.base_variant = SimpleNamespace(sequence="AAA", score_dict={"reward": 0.8},
        normalized_score_dict={"reward": 0.8}, fixed_position_indices=[], num_designable_positions=3)
    sampler.output_dir, sampler.population_size, sampler.iteration = tmp_path, 4, 0
    sampler.temperature_schedule = lambda iteration: 0.0001
    sampler.masking_schedule = lambda iteration: 1.
    sampler.rng, sampler.seed = np.random.default_rng(42), 42
    sampler.run_stats = {"num_unique_sequences": 0, "total_sequences_evaluated": 0}
    sampler.sequence_hashes, sampler.wandb_run = set(), None
    sampler.observed_value_stats = {name: dict(min=np.inf, max=-np.inf) for name in ["reward", "SCORE"]}
    return sampler


def rejection_rows(sampler):
    frame = pd.DataFrame({"sequence": ["CCC", "DDD"], "starting_sequence": ["AAA", "AAA"],
                          "starting_SCORE": [0.8, 0.8], "reward": [0.1, 0.9],
                          "parent_particle_id": [0, 1], "selected_count": [3, 1]})
    frame = sampler.calculate_scores(frame)
    return sampler.metropolis_hastings_filter(frame)


def test_rejected_child_roundtrip_reconstructs_the_actual_population(tmp_path):
    sampler = lineage_sampler(tmp_path)
    frame = rejection_rows(sampler)
    assert frame.accepted.tolist() == [False, True]
    sampler.cache_trajectory(frame)
    population, scores = sampler.update_population(frame)
    sampler._save_final_population(population, scores)
    shutdown_cache()
    saved = pd.read_csv(tmp_path / "trajectory.csv")
    assert saved.loc[0, "generated_sequence_hash"] == hash_sequence("CCC")
    assert saved.loc[0, "parent_sequence_hash"] == hash_sequence("AAA")
    assert saved.loc[0, "selected_sequence_hash"] == saved.loc[0, "sequence_hash"] == hash_sequence("AAA")
    assert saved.loc[0, "selected_SCORE"] == 0.8
    merged = load_run_data(tmp_path)
    reconstructed = merged.loc[merged.index.repeat(merged.selected_count)]
    assert reconstructed.sequence.tolist() == population == ["AAA", "AAA", "AAA", "DDD"]
    np.testing.assert_allclose(reconstructed.SCORE, scores)
    final = pd.read_csv(tmp_path / "final_population.csv")
    assert final.sequence.tolist() == population
    assert final.particle_id.tolist() == [0, 1, 2, 3]
    assert final.selected_count.sum() == sampler.population_size
    assert pd.read_csv(tmp_path / "generated_sequences.csv").sequence.tolist() == ["CCC", "DDD"]
    assert set(pd.read_csv(tmp_path / "sequence_lookup.csv").sequence) == {"AAA", "CCC", "DDD"}


def test_iteration_statistics_use_retained_states_and_multiplicities(tmp_path):
    sampler = lineage_sampler(tmp_path)
    frame = rejection_rows(sampler)
    sampler._cache_generations(frame)
    sampler._update_observed_value_stats(frame)
    logged = []
    sampler.wandb_run = SimpleNamespace(log=lambda values: logged.append(values))
    sampler._wandb_log(frame)
    assert logged[0]["reward.iteration_mean"] == pytest.approx((3 * 0.8 + 0.9) / 4)
    assert logged[0]["SCORE.iteration_mean"] == pytest.approx((3 * 0.8 + 0.9) / 4)


def test_reference_initialization_is_saved_without_fake_generated_sequences(tmp_path):
    sampler = lineage_sampler(tmp_path)
    sampler.greedy = True
    population, scores = sampler.initialize_population()
    shutdown_cache()
    merged = load_run_data(tmp_path, num_iteration_bins=1)
    assert merged.iteration.tolist() == [-1] * 4
    assert merged.parent_sequence_hash.isna().all()
    assert merged.iteration_bin.ge(0).all()
    assert merged.sequence.tolist() == population == ["AAA"] * 4
    assert merged.SCORE.tolist() == scores == [0.8] * 4
    assert sampler.run_stats["num_unique_sequences"] == 0
    assert not (tmp_path / "generated_sequences.csv").exists()


def test_loader_rejects_a_mismatched_retained_score(tmp_path):
    sampler = lineage_sampler(tmp_path)
    sampler._cache_generations(rejection_rows(sampler))
    shutdown_cache()
    trajectory = pd.read_csv(tmp_path / "trajectory.csv")
    trajectory.loc[0, "selected_SCORE"] = 0.1
    with pytest.raises(ValueError, match="retained-state scores"):
        merge_scores_into_trajectory(trajectory, pd.read_csv(tmp_path / "sequence_lookup.csv"))


def test_loader_still_reads_legacy_outputs(tmp_path):
    pd.DataFrame({"sequence": ["AAA"], "SCORE": [0.8]}).to_csv(tmp_path / "generated_sequences.csv", index=False)
    pd.DataFrame({"sequence_hash": [hash_sequence("AAA")], "selected_count": [4],
                  "accepted": [1], "iteration": [0]}).to_csv(tmp_path / "trajectory.csv", index=False)
    result = load_run_data(tmp_path)
    assert result.sequence.tolist() == ["AAA"]
    assert result.SCORE.tolist() == [0.8]


@pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
def test_complete_smc_loop_exports_the_same_population_as_saved_lineage(tmp_path, resampling_method):
    sampler = lineage_sampler(tmp_path)
    sampler.resampling_method = resampling_method
    class Generator:
        calls = 0
        def get_extra_kwargs(self, base):
            return {}
        def generate(self, frame, **kwargs):
            self.calls += 1
            return frame.assign(sequence=["AAA", "DDD", "AAA", "DDD"] if self.calls == 1 else ["CCC"] * len(frame))
    sampler.generator, sampler.generation_kwargs = Generator(), {"generations_per_sequence": 1}
    sampler.num_iters, sampler.top_k_fraction = 2, 0.2
    sampler.evaluate_variants = lambda iteration_df: iteration_df.assign(
        reward=iteration_df.sequence.map({"AAA": 0.8, "DDD": 0.9, "CCC": 0.1}))
    sampler._run()
    shutdown_cache()
    merged = load_run_data(tmp_path)
    for iteration, group in merged.groupby("iteration"):
        assert group.selected_count.sum() == sampler.population_size
    last = merged[merged.iteration == 1]
    reconstructed = last.loc[last.index.repeat(last.selected_count)]
    final = pd.read_csv(tmp_path / "final_population.csv")
    assert reconstructed.sequence.tolist() == final.sequence.tolist() == ["DDD"] * 4
    np.testing.assert_allclose(reconstructed.SCORE, final.SCORE)
    assert sampler.generator.calls == 3
