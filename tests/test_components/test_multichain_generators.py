import pytest
from pathlib import Path
import pandas as pd
from provada.components.generator import GENERATOR_REGISTRY
from provada.sequences.io import get_sequence, get_fixed_positions_from_file
from provada.base_variant import BaseVariant

# Constants
TEST_INPUTS_DIR = Path("tests/test_inputs")
COMPLEX_PDB = TEST_INPUTS_DIR / "test_complex.pdb"
COMPLEX_SEQ = TEST_INPUTS_DIR / "test_complex_seq.fasta"
COMPLEX_FIXED = TEST_INPUTS_DIR / "test_complex_fixed_positions.txt"

@pytest.fixture
def complex_sequence():
    return get_sequence(COMPLEX_SEQ)

@pytest.fixture
def masked_complex_prompt(complex_sequence):
    # complex_sequence should be "SeqA/SeqB"
    # Let's mask a region in Chain A (first chain)
    # Mask residues 10-50 in Chain A
    chains = complex_sequence.split("/")
    assert len(chains) == 2
    
    chain_a = list(chains[0])
    # Mask 5 residues
    for i in range(9, 50):
        chain_a[i] = "_"
    
    masked_a = "".join(chain_a)
    chain_b = chains[1]
    
    # Prompt: MaskedA/ChainB
    return f"{masked_a}/{chain_b}"

@pytest.mark.requires_gpu
def test_mpnn_multichain(masked_complex_prompt):
    """Test MPNN generator with multi-chain input."""
    generator = GENERATOR_REGISTRY.get_instance("mpnn")
    
    # Generate
    result = generator.generate(
        prompts=[masked_complex_prompt],
        structure_or_structures=COMPLEX_PDB,
        generations_per_sequence=2
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "sequence" in result.columns
    assert "mpnn_score" in result.columns
    
    # Verify strict structure conservation for unmasked regions
    generated_seq = result.iloc[0]["sequence"]
    gen_chains = generated_seq.split("/")
    prompt_chains = masked_complex_prompt.split("/")
    
    # Chain B should be identical (unmasked)
    assert gen_chains[1] == prompt_chains[1]
    
    # Chain A should only differ at masked positions
    for i, (char_prompt, char_gen) in enumerate(zip(prompt_chains[0], gen_chains[0])):
        if char_prompt != "_":
            assert char_prompt == char_gen

@pytest.mark.requires_gpu
def test_caliby_multichain(masked_complex_prompt):
    """Test Caliby generator with multi-chain input."""
    # Ensure caliby model loads
    generator = GENERATOR_REGISTRY.get_instance("caliby")
    
    result = generator.generate(
        prompts=[masked_complex_prompt],
        structure_or_structures=COMPLEX_PDB,
        generations_per_sequence=2,
        temperature=0.5 # ensure diverse sequence
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "sequence" in result.columns
    # Caliby returns 'caliby_energy_score' usually, based on generator.py
    assert "caliby_energy_score" in result.columns
    
    generated_seq = result.iloc[0]["sequence"]
    gen_chains = generated_seq.split("/")
    prompt_chains = masked_complex_prompt.split("/")
    
    # Chain B should be identical
    assert gen_chains[1] == prompt_chains[1]

    # Chain A unmasked matches
    for i, (char_prompt, char_gen) in enumerate(zip(prompt_chains[0], gen_chains[0])):
        if char_prompt != "_":
            assert char_prompt == char_gen

@pytest.fixture
def complex_base_variant_with_fixed_positions(complex_sequence):
    """
    Creates a BaseVariant with sequence and fixed positions loaded from file.
    Does NOT use the 'prompt' mechanism (masked sequence) but rather relies on the
    BaseVariant's fixed_indices field to drive masking/sampling logic downstream.

    However, the generator.generate() interface typically takes `prompts` (pre-masked strings) 
    OR it can take fixed_position_indices. 
    Usually the Sampler orchestrates this: Sampler -> masked prompt -> Generator.
    
    To test the generator directly with fixed positions logic, we need to manually
    replicate what the Sampler does: convert fixed positions to a mask/prompt.
    """
    variant = BaseVariant(
        sequence=complex_sequence,
        structure=COMPLEX_PDB,
        # Load the complex fixed positions: A1-A20 and Chain B
        fixed_indices=get_fixed_positions_from_file(COMPLEX_FIXED, assume_1_indexed=True)
    )
    return variant

@pytest.mark.requires_gpu
def test_multichain_fixed_positions_end_to_end(complex_base_variant_with_fixed_positions):
    """
    Tests that fixed positions from 'test_complex_fixed_positions.txt' are respected.
    This file specifies: A1-A20 fixed, and Chain B fixed (B).
    Therefore, ONLY A21+ should be mutated.
    """
    variant = complex_base_variant_with_fixed_positions
    flattened_indices = variant.get_flattened_fixed_indices()
    
    # Manually create the prompt since we aren't using the full Sampler stack here,
    # but we want to verify the logic "fixed indices -> prompt" works as expected.
    # The sampler normally does this. Let's do a simple mask based on flattened indices.
    
    sequence_arr = list(variant.sequence)
    # Mask EVERYTHING that is NOT fixed.
    # So if index i is NOT in flattened_indices AND NOT a separator, it becomes '_'
    
    for i in range(len(sequence_arr)):
        if sequence_arr[i] == '/':
            continue
        if i not in flattened_indices:
            sequence_arr[i] = '_'
            
    # Check our manual mask matches expectation
    # A1-A20 (0-19) are fixed -> Should remain original chars
    # A21+ -> Should be '_'
    # Chain B -> Should remain original chars
    prompt = "".join(sequence_arr)
    chains = prompt.split("/")
    
    # Check Chain A logic
    # First 20 residues are fixed
    assert "_" not in chains[0][:20]
    # Residue 21 onwards should be masked (unless A is shorter than 20? It's not)
    if len(chains[0]) > 20:
        # At least some should be masked. 
        # Note: if there are other rules preventing masking, this might fail, 
        # but purely based on fixed file A1-A20, the rest is "Designable".
        assert "_" in chains[0][20:]
        
    # Check Chain B logic (Fixed "B")
    # Should have NO masks
    assert "_" not in chains[1]
    
    # Now run generators with this prompt to confirm they respect it
    # Test MPNN
    mpnn = GENERATOR_REGISTRY.get_instance("mpnn")
    result_mpnn = mpnn.generate(
        prompts=[prompt],
        structure_or_structures=COMPLEX_PDB,
        generations_per_sequence=2
    )
    # Verify B is identical to original
    orig_chains = variant.sequence.split("/")
    gen_seq_mpnn = result_mpnn.iloc[0]["sequence"]
    gen_chains_mpnn = gen_seq_mpnn.split("/")
    assert gen_chains_mpnn[1] == orig_chains[1]
    # Verify A1-A20 identical
    assert gen_chains_mpnn[0][:20] == orig_chains[0][:20]

    # Test Caliby
    caliby = GENERATOR_REGISTRY.get_instance("caliby")
    result_caliby = caliby.generate(
       prompts=[prompt],
       structure_or_structures=COMPLEX_PDB,
       generations_per_sequence=2,
       temperature=0.5,
       excluded_amino_acids=["C"]
    )
    gen_seq_caliby = result_caliby.iloc[0]["sequence"]
    gen_chains_caliby = gen_seq_caliby.split("/")

    # Verify B is identical
    assert gen_chains_caliby[1] == orig_chains[1]
    # Verify A1-A20 identical
    assert gen_chains_caliby[0][:20] == orig_chains[0][:20]

    # verify that A21+-end does not have any 'C's (excluded_amino_acids test)
    assert 'C' not in gen_chains_caliby[0][20:]


