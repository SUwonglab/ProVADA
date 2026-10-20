"""
test_generators.py

Tests the generators
"""

import pytest
import pandas as pd
from provada.components.generator import GENERATOR_REGISTRY
from provada.utils.setup import seed_everything
from provada.sequences.mask import mask_k
from provada.sequences.vocab import GFP, amino_acid_sequence_check
from pathlib import Path
from provada.sequences.io import get_sequence


def helper_test_generation_function(generator, prompts, **kwargs):
    """
    Helper function that tests that calls the generation function and ensures
    expected input and output formats
    """
    # Create the input dataframe
    input_df = pd.DataFrame({"prompt": prompts})

    # Add other fields to ensure consistency with input
    input_df["other_field"] = [i for i in range(len(prompts))]

    # Deep copy the input dataframe
    input_df_copy = input_df.copy(deep=True)

    # Call the generation function
    generations = generator.generate(prompts=input_df, **kwargs)

    # Ensure input is not modified
    assert input_df.equals(input_df_copy), "Input should not be modified"

    # Ensure the output contains the "sequence" and "prompt" columns
    assert (
        "sequence" in generations.columns
    ), "Output should contain the 'sequence' column (the generated sequences)"
    assert (
        "prompt" in generations.columns
    ), "Output should contain the 'prompt' column (the original prompts)"

    # Ensure that 'other_field' is in the output
    assert (
        "other_field" in generations.columns
    ), "Output should contain all columns in the input dataframe in addition to the 'sequence' and other new columns"

    # For each value in 'other_field', ensure that the 'prompt' column is the same
    for other_field, prompt in zip(generations["other_field"], generations["prompt"]):
        assert prompt == input_df_copy.loc[other_field, "prompt"]

    # Ensure that all sequences are valid protein sequences
    for sequence in generations["sequence"].tolist():
        assert amino_acid_sequence_check(
            sequence, allow_unknown=False, allow_stop=False
        ), "Generation is not a valid sequence"

    # Return the generations
    return generations


@pytest.mark.requires_gpu
def test_esm3_generator():
    """The ESM3 generator returns one usable generation per prompt.

    What a generation looks like residue by residue — masks filled, unmasked
    positions untouched, excluded residues absent — is proto-tools' contract
    and is tested there. What is left here is the wiring on this side, which
    helper_test_generation_function covers: the input frame is not mutated, the
    extra columns survive, each row keeps its own prompt, and the result is a
    valid protein sequence.
    """
    seed_everything(42)

    prompts = [mask_k(GFP, 10) for _ in range(10)]
    esm3_generator = GENERATOR_REGISTRY.get_instance("esm3")

    generations = helper_test_generation_function(
        esm3_generator, prompts, max_batch_size=5, excluded_amino_acids=["C"]
    )

    assert len(generations) == len(prompts), "Expected one generation per prompt"


@pytest.mark.requires_gpu
@pytest.mark.parametrize("generator_name", ["mpnn", "mpnn_pytorch"])
def test_mpnn_generator(generator_name):
    """Both ProteinMPNN toolkits honour the prompt, the exclusion and the seed.

    The two implementations do not produce the same designs — they apply the
    prompt differently — so this asserts the contract both share rather than
    comparing them to each other.
    """

    test_pdb_path = Path("inputs/renin/renin_af3.pdb")
    test_sequence_path = Path("inputs/renin/example_seq_renin.txt")

    wt_seq = get_sequence(test_sequence_path)

    # Seed prior to the mask k calls
    seed_everything(42)

    # Mask a bunch of sequences
    prompts = [mask_k(wt_seq, 10) for _ in range(2)]
    # make sure all Cs are masked
    prompts = ["".join([aa if aa != "C" else "_" for aa in prompt]) for prompt in prompts]

    mpnn_generator = GENERATOR_REGISTRY.get_instance(generator_name, seed=42)

    generations_per_sequence = 10

    first_generations = helper_test_generation_function(
        mpnn_generator,
        prompts=prompts,
        structure_or_structures=test_pdb_path,
        generations_per_sequence=generations_per_sequence,
        temperature=0.2,
        excluded_amino_acids=["C", "X"],
    ).to_dict(orient="records")

    # Make list of prompts for comparision (since prompts produce multiple generations)
    all_prompts = []
    for seq in prompts:
        all_prompts.extend([seq] * 10)

    for prompt, generation in zip(all_prompts, first_generations):
        assert generation["prompt"] == prompt, "Prompt is incorrect"

    # Ensure the number of generations is correct
    assert (
        len(first_generations) == len(prompts) * generations_per_sequence
    ), "Number of generations should be equal to the number of prompts times the number of generations per prompt for MPNN"

    # ensure that no generated sequences contain omitted amino acids
    for gen in first_generations:
        for aa in "CX":
            assert aa not in gen["sequence"], f"Omitted amino acid {aa} found in generated sequence."
    
    # ensure there is no identical generatoins in the first generations
    generated_seqs = [gen["sequence"] for gen in first_generations]
    assert len(generated_seqs) == len(set(generated_seqs)), "There are identical generations in the first generations!"

    # Create a new generator with the same prompts and structure
    mpnn_generator_2 = GENERATOR_REGISTRY.get_instance(generator_name, seed=42)
    second_generations = mpnn_generator_2.generate(
        prompts=prompts,
        structure_or_structures=test_pdb_path,
        generations_per_sequence=generations_per_sequence,
        temperature=0.2,
        excluded_amino_acids=["C", "X"],
    ).to_dict(orient="records")

    # Ensure all the scores and sequences are the same
    for first_generation, second_generation in zip(first_generations, second_generations):
        assert (
            first_generation["mpnn_score"] == second_generation["mpnn_score"]
        ), "MPNN scores are not the same"
        assert (
            first_generation["sequence"] == second_generation["sequence"]
        ), "Sequences are not the same"


@pytest.mark.requires_gpu
def test_mpnn_sequence_override():
    """
    Test the MPNN generator with sequence override functionality.
    This ensures that fixed positions in the prompt are preserved in generated sequences.
    """
    seed_everything(42)

    # Get wildtype sequence from test PDB
    from provada.utils.structure import ProteinStructure

    test_pdb_path = Path("inputs/renin/renin_af3.pdb")
    structure = ProteinStructure(test_pdb_path)
    wt_seq = structure.get_chain_sequence()

    # Create a prompt with specific fixed positions
    # Let's fix positions 0, 10, 20, 30 to specific amino acids
    prompt_list = list("_" * len(wt_seq))
    fixed_positions_override = {
        0: "M",  # Position 1 (1-indexed) -> M
        10: "K",  # Position 11 -> K
        20: "P",  # Position 21 -> P
        30: "Q",  # Position 31 -> Q
    }

    for idx, aa in fixed_positions_override.items():
        prompt_list[idx] = aa

    prompt = "".join(prompt_list)

    # Generate sequences with MPNN
    mpnn_generator = GENERATOR_REGISTRY.get_instance("mpnn", seed=42)

    generations_per_sequence = 5
    generations = mpnn_generator.generate(
        prompts=[prompt],
        structure_or_structures=test_pdb_path,
        generations_per_sequence=generations_per_sequence,
        temperature=0.2,
    )

    # Verify all generated sequences preserve the fixed positions
    for _, row in generations.iterrows():
        generated_seq = row["sequence"]

        # Check that fixed positions match the prompt
        for idx, expected_aa in fixed_positions_override.items():
            actual_aa = generated_seq[idx]
            assert actual_aa == expected_aa, (
                f"Fixed position {idx} (1-indexed: {idx + 1}) should be '{expected_aa}' "
                f"but got '{actual_aa}' in sequence: {generated_seq}"
            )

        # Check that the sequence length matches
        assert len(generated_seq) == len(wt_seq), (
            f"Generated sequence length {len(generated_seq)} should match "
            f"wildtype length {len(wt_seq)}"
        )

        # Check that the prompt is preserved
        assert row["prompt"] == prompt, "Prompt should be preserved in output"

    # Verify we got the correct number of generations
    assert len(generations) == generations_per_sequence, (
        f"Should generate {generations_per_sequence} sequences, " f"but got {len(generations)}"
    )


def test_random_generator():
    """
    Test the random generator
    """

    seed_everything(42)

    prompts = [mask_k(GFP, 10) for _ in range(10)]

    random_generator = GENERATOR_REGISTRY.get_instance("random", seed=42)

    # Generate sequences
    generations = helper_test_generation_function(
        random_generator, prompts, codon_scheme="NNN"
    )

    # Check generation shape
    assert (
        len(generations) == 10
    ), "Number of generations should be equal to the number of prompts"


# ============================================================================
# Masked-model generators (ESM2 / ESM3 via proto-tools)
# ============================================================================


@pytest.mark.requires_gpu
@pytest.mark.parametrize("sampling_method", ["iterative_refinement", "single_pass"])
def test_esm2_generator(sampling_method):
    """The ESM2 generator returns one usable generation per prompt.

    Smallest checkpoint on purpose: this exercises the dispatch path, not the
    model. See test_esm3_generator for what is deliberately not asserted here.
    """
    seed_everything(42)

    prompts = [mask_k(GFP, 10) for _ in range(4)]
    esm2_generator = GENERATOR_REGISTRY.get_instance("esm2", device="cpu")

    generations = helper_test_generation_function(
        esm2_generator,
        prompts,
        max_batch_size=4,
        model_checkpoint="esm2_t6_8M_UR50D",
        sampling_method=sampling_method,
        excluded_amino_acids=["C"],
    )

    assert len(generations) == len(prompts), "Expected one generation per prompt"
