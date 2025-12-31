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

    seed_everything(42)

    prompts = [mask_k(GFP, 10) for _ in range(10)]

    esm3_generator = GENERATOR_REGISTRY.get_instance("esm3")

    generations = helper_test_generation_function(
        esm3_generator, prompts, max_batch_size=5, omit_AAs="C"
    )

    # Check generation shape
    assert (
        len(generations) == 10
    ), "Number of generations should be equal to the number of sequences"

    # Sanity check generations
    for prompt, generation in zip(prompts, generations.to_dict(orient="records")):
        for i in range(len(prompt)):
            if prompt[i] == "_":
                # If this is a designable position, make sure the generation is not a Cysteine residue
                assert (
                    generation["sequence"][i] != "C"
                ), "Generation contains Cysteine residue in masked position"

            else:
                # If this is not a designable position, make sure the generation is the same as the original sequence
                assert (
                    generation["sequence"][i] == prompt[i]
                ), "Generation is not the same as the original sequence in non-masked position"

    for prompt, generation in zip(prompts, generations.to_dict(orient="records")):
        assert generation["prompt"] == prompt, "Prompt is incorrect"


@pytest.mark.requires_gpu
def test_mpnn_generator():

    test_pdb_path = Path("inputs/renin/renin_af3.pdb")
    test_sequence_path = Path("inputs/renin/example_seq_renin.txt")

    wt_seq = get_sequence(test_sequence_path)

    # Seed prior to the mask k calls
    seed_everything(42)

    # Mask a bunch of sequences
    prompts = [mask_k(wt_seq, 10) for _ in range(2)]

    mpnn_generator = GENERATOR_REGISTRY.get_instance("mpnn", seed=42)

    generations_per_sequence = 10

    first_generations = helper_test_generation_function(
        mpnn_generator,
        prompts=prompts,
        structure_or_structures=test_pdb_path,
        generations_per_sequence=generations_per_sequence,
        temperature=0.2,
        omit_AAs="CX",
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

    # Create a new generator with the same prompts and structure
    mpnn_generator_2 = GENERATOR_REGISTRY.get_instance("mpnn", seed=42)
    second_generations = mpnn_generator_2.generate(
        prompts=prompts,
        structure_or_structures=test_pdb_path,
        generations_per_sequence=generations_per_sequence,
        temperature=0.2,
        omit_AAs="CX",
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
