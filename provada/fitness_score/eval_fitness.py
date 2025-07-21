from typing import Union, Optional
import numpy as np

from provada.utils.helpers import arr_to_aa
from provada.utils.esm_utils import (
    get_ESM_perplexity_one_pass,
    predict_location_from_seq,
)
from provada.utils.mpnn_utils import get_mpnn_scores
from provada import BaseSequenceInfo


def eval_fitness_score(
    seq: Union[str, np.ndarray],
    seq_info: BaseSequenceInfo,
    new_MPNN_score: Optional[float] = None,
    mismatch_penalty: Optional[float] = 0.0
) -> float:
    """
    Compute a fitness score for `seq` based on initial values in `seq_info`.

    Args:
        seq:               amino acid string or masked-array representing the sequence
        seq_info:          BaseSequenceInfo with initial classifier prob, perplexity, and MPNN score
        new_MPNN_score:    optional new MPNN score (if None, will be computed)
        mismatch_penalty:  penalty for mismatches (default 0.0)

    Returns:
        A scalar fitness value.
    """
    # Convert to AA string if needed
    seq_str = seq if isinstance(seq, str) else arr_to_aa(seq)

    # Classifier probability and delta
    loc_probs = predict_location_from_seq(
        seq_str,
        clf_name=seq_info.clf_name,
        classifier=seq_info.classifier,
        ESM_model=seq_info.ESM_model,
        tokenizer=seq_info.tokenizer,
        device=seq_info.device
    )

    # Map target_label to index
    if seq_info.target_label == "cytosolic":
        idx = 0
    elif seq_info.target_label == "extracellular":
        idx = 1
    else:
        raise ValueError("target_label must be 'cytosolic' or 'extracellular'")
    delta_classifier = loc_probs[idx] - seq_info.base_classifier_prob

    # Perplexity delta (non-negative)
    delta_perp = (
        get_ESM_perplexity_one_pass(
            seq=seq_str,
            model=seq_info.ESM_model,
            tokenizer=seq_info.tokenizer,
            device=seq_info.device
        ) - seq_info.base_perplexity
    )
    delta_perp = max(delta_perp, 0.0)

    # MPNN score delta (compute if needed)
    if new_MPNN_score is None:
        new_MPNN_score = get_mpnn_scores(
            pdbs=[seq_info.input_pdb],
            out_dir=seq_info.save_path,
            protein_chain=seq_info.protein_chain,
            sequence=seq_str,
            return_score=True,
            save_csv=False
        )
    delta_mpnn = new_MPNN_score - seq_info.base_MPNN_score
    delta_mpnn = max(delta_mpnn, 0.0)

    # Compute mismatch counts
    if mismatch_penalty < 1e-6:
        mismatch_frac = 0 # to save computation time
    else:
        # Count mismatches with the base sequence
        base_seq = seq_info.base_seq if isinstance(seq_info.base_seq, str) else arr_to_aa(seq_info.base_seq)
        mismatch_frac= sum(1 for a, b in zip(base_seq, seq_str) if a != b) / len(base_seq)


    # Combine with penalties
    score = (
        delta_classifier
        - delta_perp * seq_info.penalty_perplexity
        - delta_mpnn * seq_info.penalty_MPNN
        - mismatch_frac * mismatch_penalty
    )
    
    return score



def eval_fitness_scores(seqs: Union[list, np.ndarray, str],
                        seq_info: BaseSequenceInfo,
                        new_MPNN_scores: Optional[float] = None,
                        mismatch_penalty: Optional[float] = 0.0):
    """
    Evaluate fitness scores for multiple sequences.
    Args:
        seqs:              list of amino acid strings or numpy arrays, or a single sequence
        seq_info:          BaseSequenceInfo with initial classifier prob, perplexity, and MPNN score
        new_MPNN_scores:   optional new MPNN scores (if None, will be computed)
        mismatch_penalty:  penalty for mismatches (default 0.0)
    Returns:
        A list of fitness scores if multiple sequences, or a single score if one sequence.
    """

    if isinstance(seqs, str) or len(seqs.shape) == 1:
        # If only one sequence, use the single seq function
        if len(new_MPNN_scores) != 1:
            raise ValueError("new_MPNN_scores must be a single value if seqs is a single sequence")
        return eval_fitness_score(seqs, seq_info, new_MPNN_scores, mismatch_penalty)
    
    if len(seqs) != len(new_MPNN_scores):
        raise ValueError("seqs and new_MPNN_scores must have the same length")

    scores = []
    for seq, new_MPNN_score in zip(seqs, new_MPNN_scores):
        # If multiple sequences, just call the single seq function
        score = eval_fitness_score(seq, seq_info, new_MPNN_score, mismatch_penalty)
        scores.append(score)
    
    # Outputting a list of perplexities
    return scores