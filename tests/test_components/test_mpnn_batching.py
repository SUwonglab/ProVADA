"""Batch-call and ancestry regressions; no model weights or inference required."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from provada.components.generator import MPNNGenerator
from provada.sampler.sampler import MADA


@pytest.fixture
def batch_mpnn(monkeypatch):
    import provada.components.generator as module
    import provada.tools.inverse_folding.shared_data_models as models

    class Structure:
        def __init__(self, source):
            self.structure_pdb = source.structure_pdb if isinstance(source, Structure) else source["pdb"]
            self.chains = source.chains if isinstance(source, Structure) else source.get("chains", ["A"])
            self.length = source.length if isinstance(source, Structure) else source.get("length", 3)
        def get_chain_sequences(self):
            return {chain: "A" * self.length for chain in self.chains}
        def get_chain_ids(self):
            return self.chains

    monkeypatch.setattr(module, "ProteinStructure", Structure)
    monkeypatch.setattr(module, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(models, "ProteinMPNNSampleInput", SimpleNamespace)
    calls = []
    def run_sample(inputs, config):
        batch = len(calls)
        calls.append(dict(prompt=inputs.prompt, backbone=inputs.structure,
                          chains=inputs.chain_ids, size=config.num_sequences_to_generate, seed=config.seed))
        size = config.num_sequences_to_generate
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        draws = []
        for i in range(size):
            draw, value = [], i + batch * 10
            for residue in inputs.prompt:
                if residue == "_":
                    draw.append(alphabet[value % len(alphabet)])
                    value //= len(alphabet)
                else:
                    draw.append(residue)
            draws.append("".join(draw))
        calls[-1]["draws"] = draws
        return SimpleNamespace(sequences=draws,
                               scores=[float(i) / 10 for i in range(size)])
    generator = MPNNGenerator(seed=42)
    generator._tools = lambda: ((run_sample, SimpleNamespace), (None, None))
    return generator, calls, Structure


def test_one_population_sized_batch_for_identical_prompts(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    data = pd.DataFrame({"prompt": ["___"] * 128, "parent_particle_id": np.arange(128)})
    result = generator.generate(data, Structure({"pdb": "backbone"}), generations_per_sequence=1)
    assert [call["size"] for call in calls] == [128]
    assert len(result) == result.sequence.nunique() == 128
    assert result.parent_particle_id.tolist() == list(range(128))


def test_interleaved_groups_keep_row_order_and_parent_metadata(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    data = pd.DataFrame({
        "prompt": ["M__", "A__", "M__", "M__", "A__"],
        "parent_particle_id": [10, 20, 30, 40, 50],
        "starting_sequence": ["MMM", "AAA", "MCC", "MDD", "ACC"],
        "starting_SCORE": [0.1, 0.2, 0.3, 0.4, 0.5],
    }, index=[7, 7, 31, 4, 18])
    before = data.copy(deep=True)
    result = generator.generate(data, Structure({"pdb": "backbone"}), generations_per_sequence=2)
    assert [call["size"] for call in calls] == [6, 4]
    assert result.parent_particle_id.tolist() == np.repeat(data.parent_particle_id, 2).tolist()
    assert result.starting_sequence.tolist() == np.repeat(data.starting_sequence, 2).tolist()
    assert result.starting_SCORE.tolist() == np.repeat(data.starting_SCORE, 2).tolist()
    first, second = calls[0]["draws"], calls[1]["draws"]
    assert result.sequence.tolist() == first[:2] + second[:2] + first[2:4] + first[4:6] + second[2:4]
    assert result.index.tolist() == list(range(10))
    pd.testing.assert_frame_equal(data, before)


def test_same_mask_with_different_context_uses_separate_batches(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    generator.generate(["M__", "A__"], Structure({"pdb": "backbone"}), generations_per_sequence=3)
    assert [call["prompt"] for call in calls] == ["M__", "A__"]
    assert [call["size"] for call in calls] == [3, 3]


def test_same_prompt_with_different_backbones_is_not_merged(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    structures = [Structure({"pdb": pdb}) for pdb in ["backbone1", "backbone2", "backbone1"]]
    result = generator.generate(["___"] * 3, structures, generations_per_sequence=1)
    assert [(call["backbone"], call["size"]) for call in calls] == [("backbone1", 2), ("backbone2", 1)]
    assert result.sequence.tolist() == [calls[0]["draws"][0], calls[1]["draws"][0], calls[0]["draws"][1]]


def test_chain_order_is_part_of_batch_identity(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    structures = [Structure({"pdb": "backbone", "chains": chains}) for chains in [["A", "B"], ["B", "A"]]]
    generator.generate(["___/___"] * 2, structures, generations_per_sequence=1)
    assert len(calls) == 2
    assert [call["chains"] for call in calls] == [["A", "B"], ["B", "A"]]


def test_each_batch_advances_seed_while_scoring_seed_stays_fixed(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    structure = Structure({"pdb": "backbone"})
    generator.generate(["___"] * 4, structure, generations_per_sequence=2)
    generator.generate(["___"] * 4, structure, generations_per_sequence=2)
    assert [call["size"] for call in calls] == [8, 8]
    assert calls[0]["seed"] != calls[1]["seed"]
    assert generator._tool_kwargs(None)["seed"] == 42


def test_smc_initialization_dispatches_one_full_population_batch(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    sampler = MADA.__new__(MADA)
    sampler.greedy = False
    sampler.resampling_method = "systematic"
    sampler.population_size = 64
    sampler.iteration = 0
    sampler.generator = generator
    sampler.generation_kwargs = {"generations_per_sequence": 9}
    sampler.base_variant = SimpleNamespace(sequence="AAA", fixed_position_indices=[0],
                                          structure=Structure({"pdb": "backbone"}))
    sampler.run_stats = {"total_sequences_evaluated": 0}
    sampler.evaluate_variants = lambda frame: frame
    sampler.calculate_scores = lambda frame: frame.assign(SCORE=frame.mpnn_score)
    initial = []
    sampler._cache_generations = lambda frame: initial.append(frame.copy())
    population, scores = sampler.initialize_population()
    assert [(call["prompt"], call["size"]) for call in calls] == [("A__", 64)]
    assert len(population) == len(scores) == 64
    assert initial[0].parent_particle_id.tolist() == list(range(64))
    assert sampler.generation_kwargs == {"generations_per_sequence": 9}


def test_batching_preserves_smc_ancestor_weights(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    data = pd.DataFrame({"prompt": ["___"] * 3, "starting_sequence": ["AAA"] * 3,
                         "starting_SCORE": [0.0, np.log(2.0), 0.0], "parent_particle_id": [0, 1, 2]})
    generated = generator.generate(data, Structure({"pdb": "backbone"}), generations_per_sequence=2)
    generated["SCORE"] = generated.mpnn_score
    sampler = MADA.__new__(MADA)
    sampler.greedy = False
    sampler.resampling_method = "systematic"
    sampler.iteration = 0
    sampler.temperature_schedule = lambda iteration: 1.0
    sampler.rng = np.random.default_rng(42)
    selected = sampler.select_variants(generated, top_k=1, num_to_select=40)
    assert [call["size"] for call in calls] == [6]
    assert selected.groupby("parent_particle_id").selected_count.sum().to_dict() == {0: 10, 1: 20, 2: 10}


@pytest.mark.parametrize("resampling_method", ["systematic", "multinomial"])
def test_batched_smc_loop_preserves_known_target_and_call_count(batch_mpnn, resampling_method):
    # A controlled two-state prior tests the real grouping and sampler loop
    # together. This is a kernel regression, not a protein inference result.
    generator, calls, Structure = batch_mpnn
    sample, config_class = generator._tools()[0]
    def sample_prior(inputs, config):
        result = sample(inputs, config)
        rng = np.random.default_rng(config.seed)
        result.sequences = np.where(rng.random(config.num_sequences_to_generate) < 0.2, "M", "A").tolist()
        return result
    generator._tools = lambda: ((sample_prior, config_class), (None, None))
    sampler = MADA.__new__(MADA)
    sampler.greedy = False
    sampler.population_size = 1800
    sampler.resampling_method = resampling_method
    sampler.num_iters = 3
    sampler.iteration = 0
    sampler.top_k_fraction = 0.001
    sampler.rng = np.random.default_rng(42)
    sampler.temperature_schedule = lambda iteration: [2.0, 1.0, 2.0 / 3][iteration]
    sampler.masking_schedule = lambda iteration: 1.0
    sampler.generator = generator
    sampler.generation_kwargs = {"generations_per_sequence": 1}
    sampler.base_variant = SimpleNamespace(sequence="A", fixed_position_indices=[],
        normalized_score_dict={"reward": 0.0}, structure=Structure({"pdb": "backbone", "length": 1}))
    sampler.active_scores_dict = {"reward": dict(weight=1, min_value=0, max_value=1, larger_is_better=True)}
    sampler.run_stats = {"total_sequences_evaluated": 0}
    sampler._determine_min_max_ranges = lambda: None
    sampler.evaluate_variants = lambda iteration_df: iteration_df.assign(reward=(iteration_df.sequence == "M").astype(float))
    sampler._cache_generations = lambda frame: None
    trajectory = []
    sampler.cache_trajectory = lambda iteration_df: trajectory.append(iteration_df.copy())
    sampler._save_final_population = lambda *args: None
    np.random.seed(159)
    sampler._run()
    assert [call["size"] for call in calls] == [1800] * 4
    assert len({call["seed"] for call in calls}) == 4
    final = trajectory[-1]
    assert final.parent_particle_id.nunique() == 1800
    assert final.selected_count.sum() == 1800
    estimate = np.average(final.proposal_SCORE, weights=final.selected_count)
    expected = 0.2 * np.exp(1.5) / (0.8 + 0.2 * np.exp(1.5))
    assert abs(estimate - expected) < 0.04


@pytest.mark.parametrize("share, expected_calls", [(True, 2), (False, 3)])
def test_duplicate_ancestors_share_one_mask_only_for_batching_generators(monkeypatch, share, expected_calls):
    import provada.sampler.sampler as module
    calls = []
    def mask(sequence, p, fixed_indices):
        calls.append((sequence, p, fixed_indices))
        return f"{sequence}_mask{len(calls)}"
    monkeypatch.setattr(module, "mask_p", mask)
    sampler = MADA.__new__(MADA)
    sampler.generator = SimpleNamespace(batch_shared_prompts=share)
    sampler.iteration = 0
    sampler.masking_schedule = lambda iteration: 0.5
    sampler.base_variant = SimpleNamespace(fixed_position_indices=[0])
    result = sampler.mask_sequences(pd.DataFrame({"sequence": ["AAA", "MMM", "AAA"], "parent_particle_id": [4, 5, 6]}))
    assert len(calls) == expected_calls
    assert (result.prompt.iloc[0] == result.prompt.iloc[2]) == share
    assert result.parent_particle_id.tolist() == [4, 5, 6]


def test_incomplete_batch_fails_instead_of_dropping_particles(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    generator._generate = lambda **kwargs: []
    with pytest.raises(ValueError, match="batch requesting 2"):
        generator.generate(["___"] * 2, Structure({"pdb": "backbone"}), generations_per_sequence=1)
    assert calls == []


def test_structure_count_mismatch_is_rejected(batch_mpnn):
    generator, calls, Structure = batch_mpnn
    with pytest.raises(ValueError, match="Number of structures"):
        generator.generate(["___"] * 2, [Structure({"pdb": "backbone"})], generations_per_sequence=1)
    assert calls == []
