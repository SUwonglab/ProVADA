"""Calibration boundaries without model downloads or inference."""
import numpy as np
import pandas as pd
import pytest

from provada.base_variant import BaseVariant
from provada.sampler.sampler import MADA


def reference_sampler(tmp_path, sequence="AAAAAAAAAA", fixed_indices=None):
    sampler = MADA.__new__(MADA)
    sampler.base_variant = BaseVariant(sequence=sequence, fixed_indices=fixed_indices or [])
    sampler.base_variant.score_dict = {"metric": 0.5}
    sampler.active_scores_dict = {
        "metric": dict(weight=1., min_value=0., max_value="sequence_length", larger_is_better=True)
    }
    sampler.normalize = True
    sampler.wandb_run = None
    sampler.reference_distribution_file = None
    sampler.output_dir = tmp_path
    sampler.generate_variants = lambda iteration_df: iteration_df.assign(
        sequence=iteration_df.prompt.str.replace("_", "C", regex=False))
    sampler._evaluate_variants_subset = lambda iteration_df, score_subset: iteration_df.assign(metric=0.5)
    return sampler


@pytest.mark.parametrize("length,free_position", [(1, 0), (10, 4)])
def test_reference_calibration_supports_one_designable_residue(tmp_path, length, free_position):
    sampler = reference_sampler(tmp_path, sequence="A" * length,
                                fixed_indices=[i for i in range(length) if i != free_position])
    prompts = []
    def generate(iteration_df):
        prompts.extend(iteration_df.prompt)
        return iteration_df.assign(sequence=iteration_df.prompt.str.replace("_", "C", regex=False))
    sampler.generate_variants = generate
    result = sampler._get_reference_distribution([("metric", "min_value", None)], number_of_sequences=8)
    assert len(result) == 8
    assert prompts == ["A" * free_position + "_" + "A" * (length - free_position - 1)] * 8
    assert pd.read_csv(tmp_path / "reference_distribution.csv").sequence.tolist() == result.sequence.tolist()


def test_reference_calibration_includes_the_maximum_valid_mask_size(tmp_path, monkeypatch):
    sampler = reference_sampler(tmp_path, fixed_indices=list(range(8)))
    calls = []
    def randint(low, high):
        calls.append((low, high))
        return high - 1
    monkeypatch.setattr(np.random, "randint", randint)
    result = sampler._get_reference_distribution([("metric", "min_value", None)], number_of_sequences=2)
    assert calls == [(1, 3), (1, 3)]
    assert result.sequence.tolist() == ["AAAAAAAACC"] * 2


def test_reference_calibration_rejects_zero_designable_residues(tmp_path):
    sampler = reference_sampler(tmp_path, sequence="AA", fixed_indices=[0, 1])
    with pytest.raises(ValueError, match="at least one designable residue"):
        sampler._get_reference_distribution([("metric", "min_value", None)], number_of_sequences=1)
    assert not (tmp_path / "reference_distribution.csv").exists()


@pytest.mark.parametrize("normalize", [False, True])
def test_all_fixed_sampler_is_rejected_before_compute_startup(monkeypatch, normalize):
    from types import SimpleNamespace
    def forbidden_context(self):
        pytest.fail("All-fixed inputs must be rejected before acquiring workers")
    monkeypatch.setattr(MADA, "_compute_context", forbidden_context)
    finished = []
    wandb = SimpleNamespace(finish=lambda exit_code: finished.append(exit_code))
    with pytest.raises(ValueError, match="all residues are fixed"):
        MADA(base_variant=BaseVariant(sequence="AC", fixed_indices=[0, 1]),
                       num_iters=1, population_size=2, top_k_fraction=1.,
                       score_weights={"metric": 1.}, normalize=normalize, wandb_run=wandb)
    assert finished == [1]


def test_attribute_bounds_skip_empirical_calibration(tmp_path):
    sampler = reference_sampler(tmp_path)
    def forbidden_calibration(*args):
        pytest.fail("Known attribute bounds must not generate calibration sequences")
    sampler._get_reference_distribution = forbidden_calibration
    sampler._determine_min_max_ranges()
    info = sampler.active_scores_dict["metric"]
    assert info["max_value"] == 10
    assert info["specified_max_value"] == "sequence_length"
    assert sampler.base_variant.normalized_score_dict["metric"] == pytest.approx(.05)


def test_mixed_bounds_calibrate_only_unknown_metrics(tmp_path):
    sampler = reference_sampler(tmp_path)
    sampler.active_scores_dict["unknown"] = dict(
        weight=1., min_value=None, max_value=1., larger_is_better=True)
    sampler.base_variant.score_dict["unknown"] = .5
    calls = []
    def reference_distribution(request):
        assert sampler.active_scores_dict["metric"]["max_value"] == 10
        calls.append(request)
        return pd.DataFrame({"unknown": [.2, .4, .6, .8]})
    sampler._get_reference_distribution = reference_distribution
    sampler._determine_min_max_ranges()
    assert calls == [[("unknown", "min_value", None)]]
    assert sampler.active_scores_dict["unknown"]["min_value"] == pytest.approx(.35)
    assert sampler.active_scores_dict["unknown"]["max_value"] == 1.


def test_raw_fitness_skips_bound_resolution_and_calibration(tmp_path):
    sampler = reference_sampler(tmp_path)
    sampler.normalize = False
    sampler.active_scores_dict["metric"]["max_value"] = "unused_attribute"
    sampler._get_reference_distribution = lambda *args: pytest.fail("Raw fitness needs no calibration")
    sampler._determine_min_max_ranges()
    assert sampler.base_variant.normalized_score_dict == {}


@pytest.mark.parametrize("larger_is_better", [False, True])
@pytest.mark.parametrize(
    "lower,upper,baseline,scores,expected",
    [
        (None, None, 0, [0] * 7 + [3], (0., 3.)),
        (None, None, 5., [5.] * 7 + [2.], (2., 5.)),
        (0., None, 0., [0.] * 7 + [3.], (0., 3.)),
        (None, 10., 10., [10.] * 7 + [8.], (8., 10.)),
        (1., None, 1., [1.] * 7 + [0., 3.], (1., 3.)),
        (None, 9., 9., [9.] * 7 + [8., 10.], (8., 9.)),
    ],
)
def test_tied_calibration_quartiles_use_observed_spread(
    tmp_path, larger_is_better, lower, upper, baseline, scores, expected
):
    sampler = reference_sampler(tmp_path)
    sampler.base_variant.score_dict["metric"] = baseline
    info = sampler.active_scores_dict["metric"]
    info.update(min_value=lower, max_value=upper, larger_is_better=larger_is_better)
    calls = []
    def calibrate(request):
        calls.append(request)
        return pd.DataFrame({"metric": scores})
    sampler._get_reference_distribution = calibrate
    sampler._determine_min_max_ranges()
    assert len(calls) == 1
    assert (info["min_value"], info["max_value"]) == expected
    if lower is None:
        assert isinstance(info["min_value"], float)
    if upper is None:
        assert isinstance(info["max_value"], float)
    assert info["specified_min_value"] == ("None" if lower is None else lower)
    assert info["specified_max_value"] == ("None" if upper is None else upper)
    probe = sampler._fitness_component([expected[0], expected[1]], "metric", info)
    np.testing.assert_allclose(probe, [0., 1.] if larger_is_better else [1., 0.])
    sampler._fitness_component([baseline], "metric", info)
    assert len(calls) == 1


@pytest.mark.parametrize("lower,upper", [(None, None), (0., None), (None, 0.)])
def test_constant_calibration_requires_explicit_valid_bounds(tmp_path, lower, upper):
    sampler = reference_sampler(tmp_path)
    sampler.base_variant.score_dict["metric"] = 0.
    sampler.active_scores_dict["metric"].update(min_value=lower, max_value=upper)
    sampler._get_reference_distribution = lambda request: pd.DataFrame({"metric": [0.] * 8})
    with pytest.raises(ValueError, match="no calibration spread.*explicit finite bounds"):
        sampler._determine_min_max_ranges()


def test_explicit_zero_width_bounds_are_not_recalibrated(tmp_path):
    sampler = reference_sampler(tmp_path)
    sampler.active_scores_dict["metric"].update(min_value=0., max_value=0.)
    sampler._get_reference_distribution = lambda request: pytest.fail("Explicit bounds must not be refitted")
    with pytest.raises(ValueError, match="finite bounds with max_value > min_value"):
        sampler._determine_min_max_ranges()
    assert sampler.active_scores_dict["metric"]["min_value"] == 0.
    assert sampler.active_scores_dict["metric"]["max_value"] == 0.
