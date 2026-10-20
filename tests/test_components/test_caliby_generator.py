"""Tests for ProVADA's wiring into proto-tools' Caliby toolkit.

Caliby itself is covered by proto-tools' own tests, so what is checked here is
the translation ProVADA does on either side of the tool: prompt positions into
pinned residues, and designed residues back into a full-length sequence.
"""

import pytest
from pathlib import Path
from provada.components.generator import (
    GENERATOR_REGISTRY,
    chain_numbering,
    prompt_pins,
    restore_unresolved_residues,
)
from provada.utils.setup import seed_everything
from provada.utils.structure import ProteinStructure
from provada.sequences.mask import mask_k
from provada.sequences.io import get_sequence
from tests.test_components.test_generators import helper_test_generation_function

TEST_PDB_PATH = Path("tests/test_inputs/7xhz.cif")
MISSING_RESIDUE_PDB_PATH = Path("tests/test_inputs/pdb_with_missing_residues.pdb")
MISSING_RESIDUE_SEQ_PATH = Path("tests/test_inputs/test_seq_with_missing.txt")


class FakeProtoStructure:
    """Stands in for the filtered structure the tool is given, read for numbering."""

    def __init__(self, positions):
        self.positions = positions

    def get_chain_ids(self):
        return list(self.positions)

    def get_chain_positions(self, chain_id):
        return list(self.positions[chain_id])


class FakeStructure:
    """Stands in for a ProteinStructure: a chain's residues, and what it omits."""

    def __init__(self, residues, missing_residues_triplets=None):
        self.residues = residues
        self.missing_residues_triplets = missing_residues_triplets or {}

    def get_residue_position_map(self):
        return self.residues


def test_chain_numbering_covers_residues_the_model_will_not_see():
    """A chain's numbering spans every residue, designed or not."""
    # Nothing unresolved: every residue is designed.
    assert chain_numbering(
        FakeStructure({"A": [("A", 1), ("C", 2), ("D", 3)]}),
        FakeProtoStructure({"A": [1, 2, 3]}),
    ) == {"A": ([1, 2, 3], {})}

    # Residues REMARK 465 records take their place in the numbering.
    assert chain_numbering(
        FakeStructure(
            {"A": [("A", 1), ("C", 2), ("E", 5), ("F", 6)]},
            {"A": [(3, " ", "G"), (4, " ", "S")]},
        ),
        FakeProtoStructure({"A": [1, 2, 5, 6]}),
    ) == {"A": ([1, 2, 3, 4, 5, 6], {3: "G", 4: "S"})}

    # So does a residue dropped from the structure for having no alpha carbon.
    assert chain_numbering(
        FakeStructure({"A": [("Q", 1), ("C", 2), ("D", 3)]}),
        FakeProtoStructure({"A": [2, 3]}),
    ) == {"A": ([1, 2, 3], {1: "Q"})}

    # A residue recorded as missing but present in the coordinates is designed;
    # counting it twice would misplace every position after it.
    assert chain_numbering(
        FakeStructure({"A": [("A", 1), ("C", 2), ("D", 3)]}, {"A": [(2, " ", "G")]}),
        FakeProtoStructure({"A": [1, 2, 3]}),
    ) == {"A": ([1, 2, 3], {})}


def test_prompt_pins_pins_every_unmasked_position():
    """Positions the prompt does not mask are pinned to the prompt's residue."""
    numbering = {"A": ([1, 2, 3, 4], {})}
    assert prompt_pins("AC_E", numbering) == {"A": {1: "A", 2: "C", 4: "E"}}
    assert prompt_pins("____", numbering) == {}

    # Prompt segments follow structure order, retained in the numbering mapping.
    assert prompt_pins("AC/WY", {"B": ([1, 2], {}), "A": ([1, 2], {})}) == {
        "B": {1: "A", 2: "C"},
        "A": {1: "W", 2: "Y"},
    }


def test_prompt_pins_numbers_positions_among_designed_residues():
    """A gap shifts later pins, and an unresolved position is never pinned."""
    numbering = {"A": ([1, 2, 3, 4, 5, 6], {3: "G", 4: "S"})}
    # The prompt covers all six residues; 5 and 6 are the 3rd and 4th designed.
    assert prompt_pins("A_GS_F", numbering) == {"A": {1: "A", 4: "F"}}


def test_prompt_pins_rejects_a_prompt_that_does_not_fit_the_structure():
    """A prompt names every chain and spans each in full, gaps included."""
    with pytest.raises(ValueError, match="does not match number of chains"):
        prompt_pins("ACDE", {"A": ([1, 2], {}), "B": ([1, 2], {})})

    with pytest.raises(ValueError, match="without"):
        prompt_pins("ACD", {"A": ([1, 2, 3, 4], {3: "G"})})


def test_restore_unresolved_residues_puts_back_the_structures_residues():
    """Unresolved residues return at their own positions, with their own identity."""
    assert restore_unresolved_residues(
        {"A": "ACDE"}, {"A": ([1, 2, 3, 4], {})}
    ) == "ACDE"

    assert restore_unresolved_residues(
        {"A": "ACEF"}, {"A": ([1, 2, 3, 4, 5, 6], {3: "G", 4: "S"})}
    ) == "ACGSEF"

    # Chains are joined in structure order, matching the prompt.
    assert restore_unresolved_residues(
        {"A": "AC", "B": "DE"}, {"B": ([1, 2], {}), "A": ([1, 2, 3], {2: "W"})}
    ) == "DE/AWC"

    # A design that does not cover the designed residues is an error, not a slice.
    with pytest.raises(ValueError, match="with coordinates"):
        restore_unresolved_residues({"A": "AC"}, {"A": ([1, 2, 3, 4], {})})


def test_numbering_of_a_real_structure_with_missing_residues():
    """A structure's numbering covers the full sequence ProVADA designs against."""
    from proto_tools.entities.structures import Structure

    structure = ProteinStructure(MISSING_RESIDUE_PDB_PATH)
    proto_structure = Structure(
        structure=structure.resolved_structure_pdb, structure_format="pdb"
    )

    full_numbering, unresolved = chain_numbering(structure, proto_structure)["D"]
    assert len(unresolved) == 15, "the GS linker is unresolved"
    assert len(full_numbering) == len(get_sequence(MISSING_RESIDUE_SEQ_PATH))


def test_caliby_generators_are_registered():
    """Both Caliby generators resolve, and load different checkpoints."""
    assert GENERATOR_REGISTRY.get_instance("caliby")._CHECKPOINT == "caliby"
    assert (
        GENERATOR_REGISTRY.get_instance("soluble_caliby")._CHECKPOINT
        == "soluble_caliby_v1"
    ), (
        "The soluble_caliby generator must load the latest soluble_caliby_v1 "
        "checkpoint, not the older soluble_caliby one."
    )


@pytest.mark.requires_gpu
def test_caliby_generator():
    """One design per requested generation, each keeping its prompt's residues."""
    structure = ProteinStructure(TEST_PDB_PATH)
    seed_everything(0)
    prompts = [mask_k(structure.get_chain_sequence("A"), 100) for _ in range(2)]

    generations_per_sequence = 2
    results = helper_test_generation_function(
        GENERATOR_REGISTRY.get_instance("caliby", seed=0),
        prompts=prompts,
        structure_or_structures=structure,
        generations_per_sequence=generations_per_sequence,
        temperature=0.5,
    ).to_dict(orient="records")

    assert len(results) == len(prompts) * generations_per_sequence
    assert "caliby_energy_score" in results[0]

    all_prompts = []
    for prompt in prompts:
        all_prompts.extend([prompt] * generations_per_sequence)

    for prompt, generation in zip(all_prompts, results):
        assert generation["prompt"] == prompt
        assert len(generation["sequence"]) == len(prompt)
        for prompt_aa, generated_aa in zip(prompt, generation["sequence"]):
            if prompt_aa != "_":
                assert prompt_aa == generated_aa, "A pinned position was redesigned"


@pytest.mark.requires_gpu
def test_caliby_generator_scores_a_structures_own_sequence():
    """Scoring returns one Potts energy per requested row."""
    structure = ProteinStructure(TEST_PDB_PATH)
    gen = GENERATOR_REGISTRY.get_instance("caliby", seed=0)

    scores = gen.score(structure_or_structures=structure)
    assert list(scores.columns) == ["caliby_energy_score"]
    assert len(scores) == 1

    # Caliby scores the structure's sequence, so sequences only set the row count.
    native = structure.get_chain_sequence("A")
    repeated = gen.score(structure_or_structures=structure, sequences=[native, native])
    assert len(repeated) == 2
    assert repeated["caliby_energy_score"].nunique() == 1
    assert (
        repeated["caliby_energy_score"].iloc[0] == scores["caliby_energy_score"].iloc[0]
    )


@pytest.mark.requires_gpu
def test_caliby_missing_residue_handling():
    """A design keeps the unresolved residues of a structure with a missing linker."""
    structure = ProteinStructure(MISSING_RESIDUE_PDB_PATH)
    wt_seq = get_sequence(MISSING_RESIDUE_SEQ_PATH)
    missing_residues_triplets = next(iter(structure.missing_residues_triplets.values()))

    seed_everything(0)
    prompts = [mask_k(wt_seq, 10)]

    results = helper_test_generation_function(
        GENERATOR_REGISTRY.get_instance("caliby", seed=0),
        prompts=prompts,
        structure_or_structures=structure,
        generations_per_sequence=2,
        temperature=0.5,
    ).to_dict(orient="records")

    for generation in results:
        assert len(generation["sequence"]) == len(prompts[0]), (
            "Generated sequence length does not match prompt length"
        )
        for position, _icode, residue in missing_residues_triplets:
            assert generation["sequence"][position - 1] == residue, (
                "Missing residue position sequence is not the same as the input."
            )
