import pytest
from pathlib import Path
from provada.models.mpnn import ProteinMPNNModel
from provada.sequences.io import get_sequence
from provada.sequences.mask import mask_p
from provada.utils.setup import seed_everything

from provada.utils.structure import ProteinStructure

test_pdb_path = Path("inputs/renin/renin_af3.pdb")
test_sequence_path = Path("inputs/renin/example_seq_renin.txt")
test_pdb_path_missing_residues = Path("tests/test_inputs/pdb_with_missing_residues.pdb")
test_sequence_path_missing_residues = Path("tests/test_inputs/test_seq_with_missing.txt")

wt_seq = get_sequence(test_sequence_path)
wt_seq_with_missing_residues = get_sequence(test_sequence_path_missing_residues)


@pytest.mark.requires_gpu
def test_mpnn_score():
    mpnn_model = ProteinMPNNModel(seed=42)

    scores = mpnn_model.score(structure=test_pdb_path, sequences=[wt_seq])

    expected_score = 1.3162076473236084

    assert (
        scores[0] == expected_score
    ), f"MPNN score is not correct. Expected {expected_score} but got {scores[0]}"


@pytest.mark.requires_gpu
def test_mpnn_sample():

    mpnn_model = ProteinMPNNModel(seed=42)

    seed_everything(42)
    # Randomly mask 20% of the sequence by replacing 20% of the sequence with '_'
    masked_seq = mask_p(wt_seq, p=0.2)

    # Generate the generations
    generation_dict = mpnn_model.sample(
        structure=test_pdb_path,
        num_sequences_to_generate=10,
        temperature=0.2,
        fixed_positions=[
            masked_ind + 1
            for masked_ind in range(len(masked_seq))
            if masked_seq[masked_ind] != "_"
        ],
        omit_AAs="C",
    )

    assert len(generation_dict["filled_seqs"]) == 10, "Number of sequences is not correct"

    for seq in generation_dict["filled_seqs"]:
        assert len(seq) == len(wt_seq), "Sequence length is not consistent"

        for ind, (masked_aa, generated_aa) in enumerate(zip(masked_seq, seq)):
            if masked_aa != "_":
                assert (
                    masked_aa == generated_aa
                ), f"Generated sequence is different from original at position {ind}: Fixed position AA: {masked_aa} != Generated AA: {generated_aa}"

            elif masked_aa == "_":
                assert (
                    generated_aa not in "CX"
                ), f"Generated sequence contains invalid amino acids: {generated_aa} at index {ind}"


@pytest.mark.requires_gpu
def test_mpnn_sample_override():
    mpnn_model = ProteinMPNNModel(seed=42)

    seed_everything(42)
    structure_with_missing_residues = ProteinStructure(test_pdb_path_missing_residues)

    # validate structure with missing residues
    missing_residues_dict = structure_with_missing_residues._missing_residues
    expected_missing_residues = {
        "D": [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
    }
    assert (
        missing_residues_dict == expected_missing_residues
    ), f"Missing residues not identified correctly. Expected {expected_missing_residues} but got {missing_residues_dict}"

    # fix positions, ignoring missing residues
    # Randomly mask 20% of the sequence by replacing 20% of the sequence with '_', ignore missing residues
    missing_residues_indices = [i - 1 for i in missing_residues_dict.get("D", [])]
    masked_seq = mask_p(
        wt_seq_with_missing_residues, p=0.2, fixed_indices=missing_residues_indices
    )

    # Generate the generations
    generation_dict = mpnn_model.sample(
        structure=structure_with_missing_residues,
        prompt=wt_seq_with_missing_residues,
        num_sequences_to_generate=1,
        temperature=0.2,
        fixed_positions=[
            masked_ind + 1
            for masked_ind in range(len(masked_seq))
            if masked_seq[masked_ind] != "_"
        ],
        omit_AAs="C",
    )

    generated_seq = generation_dict["filled_seqs"][0]
    filled_missing_residues = generated_seq[107:122]  # residues 108-122

    # check if missing residues are handled properly
    assert (
        filled_missing_residues == wt_seq_with_missing_residues[107:122]
    ), f"Missing residues not fixed correctly. Expected {wt_seq_with_missing_residues[107:122]} but got {filled_missing_residues}"

    # check if generated sequence is unchanged
    assert (
        generated_seq != wt_seq_with_missing_residues
    ), "Generated sequence is identical to the input sequence!"

    # test if sequence override works by generating based on sequence generated above
    # fix a different set of positions
    # indices to fix: all masked positions in round 1 and missing residue
    indices_to_fix = [
        i for i, aa in enumerate(masked_seq) if aa == "_"
    ] + missing_residues_indices
    masked_seq_2 = mask_p(generated_seq, p=0.2, fixed_indices=indices_to_fix)

    generation_dict_2 = mpnn_model.sample(
        structure=structure_with_missing_residues,
        prompt=generated_seq,
        num_sequences_to_generate=1,
        temperature=0.2,
        fixed_positions=[
            ind + 1 for ind in range(len(masked_seq_2)) if masked_seq_2[ind] != "_"
        ],
        omit_AAs="C",
    )

    # check if the fixed positions are indeed fixed
    generated_seq_2 = generation_dict_2["filled_seqs"][0]
    for ind, (masked_aa, generated_aa) in enumerate(zip(masked_seq_2, generated_seq_2)):
        if masked_aa != "_":
            assert (
                masked_aa == generated_aa
            ), f"Generated sequence is different from original at position {ind}: Fixed position AA: {masked_aa} != Generated AA: {generated_aa}"
