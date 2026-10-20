"""SMC policy, finite-state target, and proposal-stream regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from provada.config import SamplerConfig
from provada.sampler.sampler import MADA, resample_systematic
from provada.sampler.schedules import Linear, PowerLaw
from provada.components.generator import MPNNGenerator, MaskedModelGenerator


def make_sampler(greedy=False, temperatures=(1.0,), resampling_method="systematic"):
    sampler = MADA.__new__(MADA)
    sampler.greedy = greedy
    sampler.resampling_method = resampling_method
    sampler.iteration = len(temperatures) - 1
    sampler.seed = 42
    sampler.rng = np.random.default_rng(42)
    sampler.temperature_schedule = lambda iteration: temperatures[iteration]
    return sampler


def candidates(parent_scores, moved_scores=None, parent_ids=None):
    if moved_scores is None:
        moved_scores = parent_scores
    size = len(parent_scores)
    result = pd.DataFrame({
        "sequence": [f"child_{i}" for i in range(size)],
        "starting_sequence": [f"parent_{i}" for i in range(size)],
        "starting_SCORE": parent_scores,
        "SCORE": moved_scores,
        "selected_count": np.zeros(size, dtype=int),
    })
    if parent_ids is not None:
        result["parent_particle_id"] = parent_ids
    return result


def test_default_is_greedy_top_k():
    config = SamplerConfig(score_weights={"dummy_score": 1.0})
    assert config.sampler_type == "mada"
    assert config.greedy is True
    assert config.top_k_fraction == 0.2
    assert config.resampling_method == "systematic"
    assert SamplerConfig(score_weights={"dummy_score": 1.0}, greedy=False).greedy is False


def test_greedy_flag_cannot_silently_modify_a_legacy_policy():
    config = SamplerConfig(score_weights={"dummy_score": 1.0}, sampler_type="rejection")
    assert config.greedy is False
    with pytest.raises(ValueError, match="greedy is an option"):
        SamplerConfig(score_weights={"dummy_score": 1.0}, sampler_type="rejection", greedy=True)


def test_stochastic_top_k_is_population_size_and_logs_ignored_fraction_once():
    sampler = make_sampler()
    sampler.population_size = 30
    sampler.top_k_fraction = 0.2
    with patch("provada.sampler.sampler.logger.info") as message:
        assert sampler._get_top_k(30) == 30
        assert sampler._get_top_k(90) == 30
    message.assert_called_once_with(
        "top_k_fraction is ignored for stochastic sampling; using K=N=30."
    )


@pytest.mark.parametrize("fraction,num_candidates", [(1.0, 30), (0.2, 150)])
def test_stochastic_top_k_matching_population_size_needs_no_message(fraction, num_candidates):
    sampler = make_sampler()
    sampler.population_size = 30
    sampler.top_k_fraction = fraction
    with patch("provada.sampler.sampler.logger.info") as message:
        assert sampler._get_top_k(num_candidates) == 30
    message.assert_not_called()


def test_greedy_top_k_keeps_fraction_of_candidate_rows_without_message():
    sampler = make_sampler(greedy=True)
    sampler.population_size = 30
    sampler.top_k_fraction = 0.2
    with patch("provada.sampler.sampler.logger.info") as message:
        assert sampler._get_top_k(30) == 6
        assert sampler._get_top_k(90) == 18
    message.assert_not_called()


def test_smc_does_not_discard_equal_weight_particles_at_top_k():
    data = candidates([0.0] * 8, list(np.arange(8, dtype=float)))
    selected = make_sampler().select_variants(data, top_k=1, num_to_select=8)
    assert selected.selected_count.tolist() == [1] * 8


@pytest.mark.parametrize("temperatures", [(1.0,), (1.0, 0.5)])
def test_incremental_weights_use_parents_not_moved_scores(temperatures):
    # Both schedules have delta-beta=1. Descendant rewards point in the
    # opposite direction, so weighting the moved states gives the wrong result.
    data = candidates([np.log(3.0), 0.0], [-10.0, 10.0])
    selected = make_sampler(temperatures=temperatures).select_variants(data, 1, 40)
    assert selected.selected_count.tolist() == [30, 10]


def test_extra_descendants_do_not_inflate_ancestor_mass():
    data = candidates([0.0] * 4, parent_ids=[0, 1, 1, 1])
    selected = make_sampler().select_variants(data, top_k=1, num_to_select=100)
    by_parent = selected.groupby("parent_particle_id").selected_count.sum()
    assert by_parent.to_dict() == {0: 50, 1: 50}


def test_greedy_true_restricts_selection_to_top_candidates():
    data = candidates([0.0] * 3, [-1.0, 0.0, 1.0])
    selected = make_sampler(greedy=True).select_variants(data, 1, 12)
    assert selected.loc[selected.selected_count > 0, "proposal_SCORE"].tolist() == [1.0]
    assert selected.selected_count.sum() == 12


def test_smc_initializes_from_prior_completions_with_fixed_residues():
    sampler = make_sampler()
    sampler.population_size = 2
    sampler.base_variant = SimpleNamespace(sequence="AMC", fixed_position_indices=[0, 2])
    sampler.run_stats = {"total_sequences_evaluated": 0}
    seen = []
    def generate(frame, **kwargs):
        assert frame.prompt.tolist() == ["A_C", "A_C"]
        assert kwargs["generations_per_sequence"] == 1
        return frame.assign(sequence=["AAC", "ACC"])
    sampler.generate_variants = generate
    sampler.evaluate_variants = lambda frame: frame
    sampler.calculate_scores = lambda frame: frame.assign(SCORE=[0.25, 0.75])
    sampler._cache_generations = lambda frame: seen.append((sampler.iteration, frame.copy()))
    population, scores = sampler.initialize_population()
    assert population == ["AAC", "ACC"]
    assert scores == [0.25, 0.75]
    assert sampler.iteration == 0
    assert sampler.run_stats["total_sequences_evaluated"] == 2
    assert seen[0][0] == -1


def test_greedy_initialization_retains_reference_start():
    sampler = make_sampler(greedy=True)
    sampler.population_size = 3
    sampler.active_scores_dict = {"reward": dict(weight=1, min_value=0, max_value=1, larger_is_better=True)}
    sampler.base_variant = SimpleNamespace(sequence="AMC", score_dict={"reward": 0.4},
                                          normalized_score_dict={"reward": 0.4})
    sampler._cache_generations = lambda *args, **kwargs: None
    assert sampler.initialize_population() == (["AMC"] * 3, [0.4] * 3)


@pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
def test_smc_recovers_known_nonuniform_two_state_target(resampling_method):
    # A controlled kernel test, not a protein experiment. The proposal is an
    # independent draw from p0=(0.8, 0.2), so reward-only MH is exactly invariant
    # for pi_beta proportional to p0 * exp(beta*f), f=(0,1).
    rng = np.random.default_rng(781)
    prior_one = 0.2
    beta_previous, beta_current = 0.4, 1.3
    def target_one(beta):
        mass = prior_one * np.exp(beta)
        return mass / (1 - prior_one + mass)
    size = 18000
    parents = (rng.random(size) < target_one(beta_previous)).astype(float)
    proposals = (rng.random(size) < prior_one).astype(float)
    data = candidates(parents, proposals, np.arange(size))
    sampler = make_sampler(temperatures=(1 / beta_previous, 1 / beta_current),
                           resampling_method=resampling_method)
    np.random.seed(781)
    selected = sampler.select_variants(data, top_k=1, num_to_select=size)
    estimate = np.average(selected.proposal_SCORE, weights=selected.selected_count)
    assert abs(estimate - target_one(beta_current)) < 0.015


@pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
def test_smc_loop_tracks_known_target_over_several_temperatures(resampling_method):
    # Exercise initialization, descendant metadata, scoring, MH, resampling
    # and population updates together using a fully specified two-state prior.
    rng = np.random.default_rng(123)
    class PriorGenerator:
        def get_extra_kwargs(self, base):
            return {}
        def generate(self, prompts, generations_per_sequence=2):
            rows = []
            for _, row in prompts.iterrows():
                for _ in range(generations_per_sequence):
                    rows.append({**row.to_dict(), "sequence": "M" if rng.random() < 0.2 else "A"})
            return pd.DataFrame(rows)
    sampler = make_sampler(temperatures=(2.0, 1.0, 2.0 / 3),
                           resampling_method=resampling_method)
    sampler.iteration = 0
    sampler.population_size = 1600
    sampler.num_iters = 3
    sampler.top_k_fraction = 0.001
    sampler.base_variant = SimpleNamespace(sequence="A", fixed_position_indices=[],
        normalized_score_dict={"reward": 0.0})
    sampler.generator = PriorGenerator()
    sampler.generation_kwargs = {"generations_per_sequence": 2}
    sampler.masking_schedule = lambda iteration: 1.0
    sampler.active_scores_dict = {"reward": dict(weight=1, min_value=0, max_value=1, larger_is_better=True)}
    sampler.run_stats = {"total_sequences_evaluated": 0}
    sampler._determine_min_max_ranges = lambda: None
    sampler.evaluate_variants = lambda iteration_df: iteration_df.assign(reward=(iteration_df.sequence == "M").astype(float))
    initial = []
    sampler._cache_generations = lambda frame: initial.append(frame.copy())
    trajectory = []
    sampler.cache_trajectory = lambda iteration_df: trajectory.append(iteration_df.copy())
    sampler._save_final_population = lambda *args: None
    np.random.seed(123)
    sampler._run()
    assert len(initial) == 1
    assert sampler.iteration == 3
    assert sampler.run_stats["total_sequences_evaluated"] == 1600 + 3 * 3200
    assert len(trajectory) == 3
    final = trajectory[-1]
    assert final.parent_particle_id.nunique() == 1600
    assert final.selected_count.sum() == 1600
    estimate = np.average(final.proposal_SCORE, weights=final.selected_count)
    expected = 0.2 * np.exp(1.5) / (0.8 + 0.2 * np.exp(1.5))
    assert abs(estimate - expected) < 0.04


def test_mh_does_not_overflow_for_large_positive_log_ratio():
    sampler = make_sampler(temperatures=(1e-9,))
    with np.errstate(over="raise"):
        selected = sampler.metropolis_hastings_filter(candidates([0.0], [1.0]))
    assert selected.accepted.tolist() == [True]


@pytest.mark.parametrize("schedule", [Linear, PowerLaw])
def test_single_iteration_smc_schedule_has_a_defined_start(schedule):
    kwargs = {"alpha": 3.0} if schedule is PowerLaw else {}
    assert schedule(total_iters=1, start_value=1.0, stop_value=0.1, **kwargs)(0) == 1.0


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 2.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_systematic_resampling_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        resample_systematic(np.arange(2), weights, 5, np.random.default_rng(42))


def test_systematic_zero_offset_skips_zero_mass():
    rng = SimpleNamespace(random=lambda: 0.0)
    assert resample_systematic(np.array([0, 1]), [0.0, 1.0], 4, rng).tolist() == [1] * 4


def test_systematic_resampling_keeps_requested_size_and_seed_replay():
    idx = np.arange(3)
    first = resample_systematic(idx, [0.1, 0.3, 0.6], 103, np.random.default_rng(42))
    second = resample_systematic(idx, [0.1, 0.3, 0.6], 103, np.random.default_rng(42))
    assert len(first) == 103
    assert np.array_equal(first, second)
    assert set(first) == {0, 1, 2}


@pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
def test_smc_resampling_replays_seed_and_preserves_ancestor_mass(resampling_method):
    # One ancestor has one descendant and the other has three. Offspring count
    # must not change the intended 3:1 parent mass under either resampler.
    data = candidates([np.log(3.), 0., 0., 0.], parent_ids=[0, 1, 1, 1])
    def select():
        return make_sampler(resampling_method=resampling_method).select_variants(
            data, top_k=1, num_to_select=20000
        )
    first, second = select(), select()
    assert first.selected_count.sum() == 20000
    assert first.selected_count.tolist() == second.selected_count.tolist()
    by_parent = first.groupby("parent_particle_id").selected_count.sum()
    assert abs(by_parent[0] / 20000 - .75) < .01
    assert abs(by_parent[1] / 20000 - .25) < .01


def test_multinomial_option_uses_weighted_draws_with_replacement():
    sampler = make_sampler(resampling_method="multinomial")
    draws = []
    def choice(indices, size, replace, p):
        draws.append((indices, size, replace, p))
        return np.array([0, 0, 0, 0, 0, 0, 1, 1])
    sampler.rng = SimpleNamespace(choice=choice)
    result = sampler.select_variants(candidates([np.log(3.), 0.]), 1, 8)
    assert len(draws) == 1
    indices, size, replace, probabilities = draws[0]
    assert indices.tolist() == [0, 1]
    assert size == 8 and replace is True
    np.testing.assert_allclose(probabilities, [.75, .25])
    assert result.selected_count.tolist() == [6, 2]


def test_unknown_resampling_method_is_rejected_before_startup():
    with pytest.raises(ValueError, match="resampling_method must be"):
        MADA(base_variant=None, num_iters=1, population_size=1,
                       top_k_fraction=1., score_weights={"dummy_score": 1.},
                       resampling_method="unknown")


def test_mpnn_proposals_advance_seed_but_scoring_seed_is_fixed(monkeypatch):
    import provada.components.generator as module
    import provada.tools.inverse_folding.shared_data_models as models
    class Structure:
        def __init__(self, source):
            self.structure_pdb = "backbone"
        def get_chain_sequences(self):
            return {"A": "AAA"}
        def get_chain_ids(self):
            return ["A"]
    monkeypatch.setattr(module, "ProteinStructure", Structure)
    monkeypatch.setattr(module, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(models, "ProteinMPNNSampleInput", SimpleNamespace)
    calls = []
    def run_sample(inputs, config):
        calls.append(config.seed)
        return SimpleNamespace(sequences=["AAA"] * config.num_sequences_to_generate,
                               scores=[1.0] * config.num_sequences_to_generate)
    def stream():
        generator = MPNNGenerator(seed=42)
        generator._tools = lambda: ((run_sample, SimpleNamespace), (None, None))
        generator.generate(["___", "___"], Structure(None), generations_per_sequence=1)
        generator.generate(["___"], Structure(None), generations_per_sequence=1)
        assert generator._tool_kwargs(None)["seed"] == 42
    stream()
    first = calls[:]
    calls.clear()
    stream()
    assert len(set(first)) == 2
    assert calls == first


def test_masked_model_requests_advance_seed_and_replay():
    class Generator(MaskedModelGenerator):
        def _generate(self, prompts, **kwargs):
            pass
    generator = Generator(seed=42)
    seeds = []
    def run_tool(inputs, config):
        seeds.append(config.seed)
        return SimpleNamespace(results=[SimpleNamespace(sequence="AAA")])
    # An explicit device avoids device discovery or any model/tool invocation.
    for _ in range(3):
        generator._dispatch(run_tool, SimpleNamespace, SimpleNamespace, ["___"], "cpu")
    assert len(set(seeds)) == 3
    expected = np.random.default_rng(42).integers(0, 2**31 - 1, size=3).tolist()
    assert seeds == expected
