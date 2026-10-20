"""Canonical scoring, fixed normalization, and missing-metric regressions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from provada.sampler.sampler import MADA
from provada.components.generator import MPNNGenerator


def fitness_sampler(normalize=True):
    sampler = MADA.__new__(MADA)
    sampler.normalize = normalize
    sampler.resampling_method = "systematic"
    sampler.active_scores_dict = {
        "reward": dict(weight=2., min_value=0., max_value=10., larger_is_better=True),
        "mpnn_score": dict(weight=1., min_value=0., max_value=3., larger_is_better=False),
    }
    sampler.base_variant = SimpleNamespace(
        sequence="AAA", score_dict={"reward": 5., "mpnn_score": 1.5},
        normalized_score_dict={"reward": 0.5, "mpnn_score": 0.5},
        structure=SimpleNamespace(structure_pdb="backbone-one", get_chain_ids=lambda: ["A"]))
    return sampler


def test_fixed_bounds_produce_oriented_unit_interval_fitness():
    sampler = fitness_sampler()
    result = sampler.calculate_scores(pd.DataFrame({
        "reward": [-10., 0., 5., 10., 20.],
        "mpnn_score": [4., 3., 1.5, 0., -1.],
    }))
    np.testing.assert_allclose(result.SCORE, [0., 0., 0.5, 1., 1.])
    np.testing.assert_allclose(result.SCORE_delta, [-0.5, -0.5, 0., 0.5, 0.5])
    assert result.SCORE.between(0, 1).all()


def test_normalization_does_not_depend_on_batch_composition():
    sampler = fitness_sampler()
    single = sampler.calculate_scores(pd.DataFrame({"reward": [7.], "mpnn_score": [0.9]}))
    batch = sampler.calculate_scores(pd.DataFrame({"reward": [-5., 7., 40.], "mpnn_score": [8., 0.9, -4.]}))
    assert single.SCORE.iloc[0] == batch.SCORE.iloc[1]


def test_raw_fitness_orients_metrics_and_reports_reference_improvement():
    sampler = fitness_sampler(normalize=False)
    # Raw scoring does not require min/max ranges.
    for info in sampler.active_scores_dict.values():
        info["min_value"], info["max_value"] = None, None
    result = sampler.calculate_scores(pd.DataFrame({"reward": [5., 8.], "mpnn_score": [1.5, 0.6]}))
    np.testing.assert_allclose(result.SCORE, [(10 - 1.5) / 3, (16 - 0.6) / 3])
    np.testing.assert_allclose(result.SCORE_delta, [0., (6 + 0.9) / 3])
    assert "reward_normalized" not in result


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, None, "missing"])
def test_nonfinite_or_nonnumeric_active_metrics_raise(bad):
    sampler = fitness_sampler()
    with pytest.raises(ValueError, match="mpnn_score"):
        sampler.calculate_scores(pd.DataFrame({"reward": [5.], "mpnn_score": [bad]}))


def test_absent_active_metric_raises_instead_of_zero_reward():
    with pytest.raises(ValueError, match="Missing active score 'mpnn_score'"):
        fitness_sampler().calculate_scores(pd.DataFrame({"reward": [5.]}))


@pytest.mark.parametrize("lower,upper", [(1., 1.), (2., 1.), (np.nan, 3.), (0., np.inf), (None, 3.)])
def test_invalid_fixed_bounds_raise(lower, upper):
    sampler = fitness_sampler()
    sampler.active_scores_dict["reward"].update(min_value=lower, max_value=upper)
    with pytest.raises(ValueError, match="finite bounds"):
        sampler.calculate_scores(pd.DataFrame({"reward": [5.], "mpnn_score": [1.5]}))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_invalid_weights_raise(bad):
    sampler = fitness_sampler()
    sampler.active_scores_dict["reward"]["weight"] = bad
    with pytest.raises(ValueError, match="weights must be finite"):
        sampler.calculate_scores(pd.DataFrame({"reward": [5.], "mpnn_score": [1.5]}))


def test_large_finite_weights_do_not_overflow_their_total():
    sampler = fitness_sampler()
    for info in sampler.active_scores_dict.values():
        info["weight"] = 1e308
    result = sampler.calculate_scores(pd.DataFrame({"reward": [5.], "mpnn_score": [1.5]}))
    assert result.SCORE.iloc[0] == 0.5


def test_reference_and_candidates_use_identical_clipping():
    sampler = fitness_sampler()
    sampler.base_variant.score_dict = {"reward": -10., "mpnn_score": 5.}
    sampler.wandb_run = None
    sampler._determine_min_max_ranges()
    result = sampler.calculate_scores(pd.DataFrame([sampler.base_variant.score_dict]))
    assert result.SCORE.iloc[0] == 0.
    assert result.SCORE_delta.iloc[0] == 0.


def test_raw_mode_skips_reference_distribution_generation():
    sampler = fitness_sampler(normalize=False)
    sampler.active_scores_dict["reward"].update(min_value=None, max_value=None)
    sampler._get_reference_distribution = lambda *a, **k: pytest.fail("raw mode requested calibration")
    sampler._determine_min_max_ranges()
    assert sampler.base_variant.normalized_score_dict == {}


def test_nonfinite_reference_metric_raises_before_range_calibration():
    sampler = fitness_sampler()
    sampler.base_variant.score_dict["mpnn_score"] = np.nan
    with pytest.raises(ValueError, match="mpnn_score reference"):
        sampler._determine_min_max_ranges()


def canonical_sampler():
    sampler = fitness_sampler()
    calls = []
    generator = SimpleNamespace(seed=42, _TOOLKIT="jax", _WEIGHTS="original", device="cpu")
    generator.get_extra_kwargs = lambda base: {"structure_or_structures": base.structure}

    def score(sequences, structure_or_structures):
        calls.append((sequences[:], structure_or_structures.structure_pdb, generator.seed))
        return pd.DataFrame({"mpnn_score": [sequence.count("A") / 2 for sequence in sequences]})

    generator.score = score
    sampler.generator = generator
    return sampler, calls


def test_canonical_scores_ignore_prompt_scores_and_cache_unique_sequences():
    sampler, calls = canonical_sampler()
    first = sampler._canonical_generator_scores(pd.DataFrame({
        "sequence": ["AAA", "CCC", "AAA"], "prompt": ["___", "C__", "A__"],
        "mpnn_score": [99., -99., 12.], "starting_SCORE": [0.1, 0.2, 0.3],
    }, index=[7, 2, 9]))
    second = sampler._canonical_generator_scores(pd.DataFrame({
        "sequence": ["CCC", "AAA", "AAC"], "mpnn_score": [8., 9., 10.]}))
    assert calls == [(["AAA", "CCC"], "backbone-one", 42), (["AAC"], "backbone-one", 42)]
    assert first.mpnn_score.tolist() == [1.5, 0., 1.5]
    assert first.generation_mpnn_score.tolist() == [99., -99., 12.]
    assert first.index.tolist() == [7, 2, 9]
    assert first.starting_SCORE.tolist() == [0.1, 0.2, 0.3]
    assert second.mpnn_score.tolist() == [0., 1.5, 1.]


@pytest.mark.parametrize("change", ["backbone", "seed", "weights", "chains"])
def test_canonical_cache_separates_scoring_contexts(change):
    sampler, calls = canonical_sampler()
    frame = pd.DataFrame({"sequence": ["AAA"]})
    sampler._canonical_generator_scores(frame)
    if change == "backbone":
        sampler.base_variant.structure.structure_pdb = "backbone-two"
    elif change == "seed":
        sampler.generator.seed = 17
    elif change == "weights":
        sampler.generator._WEIGHTS = "soluble"
    else:
        sampler.base_variant.structure.get_chain_ids = lambda: ["B", "A"]
    sampler._canonical_generator_scores(frame)
    assert len(calls) == 2


@pytest.mark.parametrize("result", [pd.DataFrame({"mpnn_score": [np.nan]}),
                                  pd.DataFrame({"mpnn_score": []}),
                                  pd.DataFrame({"wrong": [1.]})])
def test_invalid_canonical_scores_fail_without_poisoning_cache(result):
    sampler, _ = canonical_sampler()
    sampler.generator.score = lambda **kwargs: result
    with pytest.raises(ValueError, match="mpnn_score|MPNN scorer"):
        sampler._canonical_generator_scores(pd.DataFrame({"sequence": ["AAA"]}))
    assert sampler._generator_score_cache == {}


def test_reference_initialization_uses_the_configured_sequence_and_cache():
    sampler, calls = canonical_sampler()
    sampler.active_scores_dict = {"mpnn_score": dict(weight=1.)}
    sampler.seed = 42
    class Evaluator:
        active_scores = ["mpnn_score"]
        def get_extra_kwargs(self, base):
            return {}
        def evaluate(self, frame, **kwargs):
            return frame[["mpnn_score"]]
    sampler.evaluators = [Evaluator()]
    reference = sampler.evaluate_variants(pd.DataFrame({"sequence": ["AAA"]}))
    candidate = sampler.evaluate_variants(pd.DataFrame({"sequence": ["AAA"], "mpnn_score": [99.]}))
    assert reference.mpnn_score.iloc[0] == candidate.mpnn_score.iloc[0] == 1.5
    assert len(calls) == 1


def test_actual_mpnn_score_wrapper_uses_fixed_seed_and_explicit_sequences(monkeypatch):
    import provada.components.generator as module
    import provada.tools.inverse_folding.shared_data_models as models
    class Structure:
        def __init__(self, source):
            self.structure_pdb = "native-backbone-with-different-sequence"
        def get_chain_ids(self):
            return ["A"]
    monkeypatch.setattr(module, "ProteinStructure", Structure)
    monkeypatch.setattr(models, "ProteinMPNNScoreInput", SimpleNamespace, raising=False)
    generator = MPNNGenerator(seed=42, device="cpu")
    calls = []
    def run_score(inputs, config):
        assert inputs.sequences is not None, "scored the native backbone instead of requested sequences"
        calls.append((inputs.sequences[:], config.seed))
        return SimpleNamespace(scores=[sequence.count("A") / 2 for sequence in inputs.sequences])
    generator._tools = lambda: ((None, None), (run_score, SimpleNamespace))
    sampler = fitness_sampler()
    sampler.generator = generator
    sampler.base_variant.structure = Structure(None)
    reference = sampler._canonical_generator_scores(pd.DataFrame({"sequence": ["AAA"]}))
    first = sampler._canonical_generator_scores(pd.DataFrame({"sequence": ["AAA", "CCC", "AAA"],
                                                              "mpnn_score": [11., 12., 13.]}))
    second = sampler._canonical_generator_scores(pd.DataFrame({"sequence": ["CCC", "AAA"]}))
    assert calls == [(["AAA"], 42), (["CCC"], 42)]
    assert reference.mpnn_score.iloc[0] == first.mpnn_score.iloc[0] == second.mpnn_score.iloc[1] == 1.5


def test_shift_to_unit_interval_preserves_smc_selection_and_mh_decisions():
    def select(shift):
        sampler = fitness_sampler()
        sampler.greedy, sampler.iteration = False, 1
        sampler.temperature_schedule = lambda iteration: [2., 0.5][iteration]
        sampler.rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "sequence": ["A", "B", "C"], "starting_sequence": ["X", "Y", "Z"],
            "SCORE": np.array([0.1, -0.2, 0.3]) + shift,
            "starting_SCORE": np.array([-0.1, 0.2, 0.]) + shift,
            "parent_particle_id": [0, 1, 2], "selected_count": [0, 0, 0],
        })
        np.random.seed(37)
        return sampler.select_variants(frame, 1, 60)
    old, new = select(0.), select(0.5)
    assert old.selected_count.tolist() == new.selected_count.tolist()
    assert old.accepted.tolist() == new.accepted.tolist()


def test_systematic_smc_keeps_signed_bridge_weights_during_reheating():
    sampler = fitness_sampler()
    sampler.greedy, sampler.iteration = False, 1
    sampler.temperature_schedule = lambda iteration: [0.5, 2.][iteration]
    sampler.rng = np.random.default_rng(42)
    frame = pd.DataFrame({"sequence": ["A", "B"], "starting_sequence": ["A", "B"],
                          "SCORE": [0., np.log(3.) / 1.5],
                          "starting_SCORE": [0., np.log(3.) / 1.5],
                          "parent_particle_id": [0, 1], "selected_count": [0, 0]})
    result = sampler.select_variants(frame, 1, 40)
    assert result.selected_count.tolist() == [30, 10]
