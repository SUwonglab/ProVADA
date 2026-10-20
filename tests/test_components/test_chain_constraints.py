"""Structure-order constraints and explicit-reference scoring regressions."""
import sys
from types import ModuleType, SimpleNamespace

import gemmi
import numpy as np
import pandas as pd
import pytest

from provada.base_variant import BaseVariant
from provada.components.generator import (
    MPNNGenerator, CalibyGenerator, _parse_fixed_positions,
    chain_numbering, prompt_pins, restore_unresolved_residues,
)
from provada.sequences.io import get_sequence
from provada.utils.structure import ProteinStructure
from provada.sampler.sampler import MADA


def make_ba_structure(tmp_path):
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for chain_id, residues in [("B", ["ALA", "CYS", "ASP"]), ("A", ["GLY", "SER"])]:
        chain = gemmi.Chain(chain_id)
        for position, name in enumerate(residues, 1):
            residue = gemmi.Residue()
            residue.name, residue.seqid = name, gemmi.SeqId(position, " ")
            atom = gemmi.Atom()
            atom.name, atom.element = "CA", gemmi.Element("C")
            atom.pos = gemmi.Position(position * 3., 0., 0.)
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    path = tmp_path / "ba.pdb"
    structure.write_pdb(str(path))
    return ProteinStructure(path), path


def test_real_structure_fasta_and_constraints_follow_b_a_order(tmp_path):
    structure, path = make_ba_structure(tmp_path)
    assert structure.get_chain_ids() == ["B", "A"]
    fasta = tmp_path / "chains.fasta"
    fasta.write_text(">A\nGS\n>B\nACD\n")
    assert get_sequence(fasta, chain_ids=["B", "A"]) == "ACD/GS"
    fixed = tmp_path / "fixed.txt"
    fixed.write_text("B2 A")
    variant = BaseVariant()
    variant.load_base_variant_from_config(SimpleNamespace(
        structure_file_path=str(path), sequence_file_path=str(fasta),
        ensemble_structure_dir=None, main_conformer=None,
        fixed_position_files=[str(fixed)], mhc_alleles=None, exclude_self_peptides=False))
    assert variant.sequence == "ACD/GS"
    assert variant.get_flattened_fixed_indices() == [1, 3, 4, 5]
    prompt = "".join(residue if i in variant.fixed_position_indices else "_"
                     for i, residue in enumerate(variant.sequence))
    assert prompt == "_C_/GS"
    assert _parse_fixed_positions(prompt, structure) == {"B": [2], "A": [1, 2]}


@pytest.mark.parametrize("contents", [">A\nGS\n>A\nACD\n", ">A\nGS\n>C\nACD\n"])
def test_ambiguous_or_unknown_fasta_chains_fail(tmp_path, contents):
    fasta = tmp_path / "chains.fasta"
    fasta.write_text(contents)
    with pytest.raises(ValueError, match="chain"):
        get_sequence(fasta, chain_ids=["B", "A"])


@pytest.mark.parametrize("fixed", [{"Z": [0]}, {"B": [-1]}, {"B": [3]}, {"A": [2]}, [-1], [6]])
def test_invalid_fixed_constraints_fail(tmp_path, fixed):
    structure, _ = make_ba_structure(tmp_path)
    variant = BaseVariant(sequence="ACD/GS", structure=structure, fixed_indices=fixed)
    with pytest.raises(ValueError, match="Fixed"):
        variant.get_flattened_fixed_indices()


def test_text_segments_use_structure_order_and_all_chain_constraints(tmp_path):
    structure, _ = make_ba_structure(tmp_path)
    text = tmp_path / "seq.txt"
    text.write_text("ACD/GS")
    variant = BaseVariant(structure=structure, fixed_indices={"B": "ALL", "A": [1]})
    variant.load_sequence(str(text))
    assert variant.get_flattened_fixed_indices() == [0, 1, 2, 3, 5]
    with pytest.raises(ValueError, match="Prompt chain B"):
        _parse_fixed_positions("__/__", structure)


def test_caliby_numbering_and_roundtrip_preserve_structure_order():
    structure = SimpleNamespace(get_residue_position_map=lambda: {"B": [("A", 1), ("C", 2), ("D", 3)],
                                                                "A": [("G", 1), ("S", 2)]},
                                missing_residues_triplets={})
    proto = SimpleNamespace(get_chain_ids=lambda: ["A", "B"],
                            get_chain_positions=lambda chain: [1, 2] if chain == "A" else [1, 2, 3])
    numbering = chain_numbering(structure, proto)
    assert list(numbering) == ["B", "A"]
    assert prompt_pins("_C_/GS", numbering) == {"B": {2: "C"}, "A": {1: "G", 2: "S"}}
    assert restore_unresolved_residues({"A": "GS", "B": "ACD"}, numbering) == "ACD/GS"


def test_unresolved_residue_counts_and_whole_chain_constraints(tmp_path):
    _, path = make_ba_structure(tmp_path)
    lines = [line for line in path.read_text().splitlines()
             if not (line.startswith("ATOM") and line[21] == "B" and int(line[22:26]) == 2)]
    path.write_text("REMARK 465     CYS B     2\n" + "\n".join(lines) + "\n")
    structure = ProteinStructure(path)
    assert structure.get_chain_sequences()["B"] == "A-D"
    assert _parse_fixed_positions("ACD/GS", structure) == {"B": [1, 2, 3], "A": [1, 2]}
    variant = BaseVariant(sequence="ACD/GS", structure=structure, fixed_indices={"B": "ALL", "A": [np.int64(1)]})
    assert variant.get_flattened_fixed_indices() == [0, 1, 2, 3, 5]


def test_mpnn_backend_receives_structure_order_constraints_and_enforces_them(tmp_path, monkeypatch):
    import provada.tools.inverse_folding.shared_data_models as models
    monkeypatch.setattr(models, "ProteinMPNNSampleInput", SimpleNamespace)
    structure, _ = make_ba_structure(tmp_path)
    generator = MPNNGenerator(seed=42, device="cpu")
    calls = []
    def sample(inputs, config):
        calls.append(inputs)
        return SimpleNamespace(sequences=["ACD/GS"], scores=[1.])
    generator._tools = lambda: ((sample, SimpleNamespace), (None, None))
    result = generator.generate(["_C_/GS"], structure, generations_per_sequence=1)
    assert result.sequence.tolist() == ["ACD/GS"]
    assert calls[0].chain_ids == ["B", "A"]
    assert calls[0].fixed_positions == {"B": [2], "A": [1, 2]}
    generator._tools = lambda: ((lambda *args: SimpleNamespace(sequences=["AAA/GS"], scores=[1.]), SimpleNamespace), (None, None))
    with pytest.raises(ValueError, match="unmasked"):
        generator.generate(["_C_/GS"], structure, generations_per_sequence=1)


def test_caliby_rejects_wrong_reference_even_with_one_chain(monkeypatch):
    tool = ModuleType("proto_tools.tools.inverse_folding.caliby")
    for name in ["CalibyScoreConfig", "CalibyScoreInput", "CalibyScoreStructureInput"]:
        setattr(tool, name, SimpleNamespace)
    class Score:
        sequence = "ACD"
        def __getitem__(self, key):
            return -3.
    tool.run_caliby_score = lambda *args: SimpleNamespace(scores=[Score()])
    monkeypatch.setitem(sys.modules, tool.__name__, tool)
    generator = CalibyGenerator(seed=42, device="cpu")
    generator._ensemble = lambda ensemble: ([object()], {"A": ([1, 2, 3], {})})
    # Ensemble mode avoids any structure conversion at this model boundary.
    generator._config = lambda *args, **kwargs: SimpleNamespace()
    generator_input = SimpleNamespace()
    import provada.components.generator as module
    monkeypatch.setattr(module, "as_ensemble", lambda source: [generator_input])
    matching = generator.score(generator_input, sequences=["ACD"], is_ensemble=True)
    assert matching.caliby_energy_score.tolist() == [-3.]
    with pytest.raises(ValueError, match="requested sequence differs"):
        generator.score(generator_input, sequences=["AAA"], is_ensemble=True)


def test_startup_passes_the_actual_reference_and_propagates_scoring_failure(monkeypatch):
    class Evaluator:
        def __init__(self, seed, active_scores):
            self.active_scores = active_scores
        @classmethod
        def available_scores(cls):
            return {"caliby_energy_score": dict(min_value=-1000, max_value=0, larger_is_better=False)}
        def get_extra_kwargs(self, base):
            return {}
        def evaluate(self, frame, **kwargs):
            return pd.DataFrame({"caliby_energy_score": [np.nan] * len(frame)})
    sampler = MADA.__new__(MADA)
    sampler.seed = 42
    sampler.active_scores_dict = {"caliby_energy_score": dict(weight=1.)}
    sampler.base_variant = SimpleNamespace(sequence="AAA", structure=object())
    requests = []
    def reject_reference(**kwargs):
        requests.append(kwargs)
        raise ValueError("requested sequence differs")
    sampler.generator = SimpleNamespace(score=reject_reference,
        get_extra_kwargs=lambda base: {"structure_or_structures": base.structure, "is_ensemble": True})
    monkeypatch.setitem(sampler.initialize_evaluators.__func__.__globals__,
                        "get_evaluators_from_score_keys", lambda names: [Evaluator])
    with pytest.raises(ValueError, match="requested sequence differs"):
        sampler.initialize_evaluators()
    assert requests[0]["sequences"] == ["AAA"]
    assert requests[0]["is_ensemble"] is True
