# provada/utils/__init__.py
"""
Utility subpackage for provada: helpers, MPNN workflows, and ESM-based tools.
"""

# helper functions
from .helpers import (
    aa_to_arr,
    arr_to_aa,
    parse_masked_seq,
    get_csv,
    masked_seq_arr_to_str,
    get_sequence,
    generate_masked_seq_str,
    generate_masked_seq_arr,
    get_masked_positions,
    compute_diversity_metrics_str,
    compute_diversity_metrics_int,
    append_csv_line,
    get_mismatch_fraction_multiseqs,
    generate_masked_seqs_str,
    read_fixed_positions
)

# MPNN workflow functions
from .mpnn_utils import (
    run_mpnn,
    get_mpnn_scores,
    mpnn_masked_gen,
    fill_mpnn,
)

# ESM-based utilities
from .esm_utils import (
    init_ESM,
    get_embedding_single,
    predict_location_from_emb,
    predict_location_from_seq,
    get_ESM_perplexity_one_pass,
    predict_location_from_seqs,
    get_ESM_perplexity_one_pass_multiseqs
)

__all__ = [
    # helpers
    "aa_to_arr",
    "arr_to_aa",
    "parse_masked_seq",
    "get_csv",
    "masked_seq_arr_to_str",
    "get_sequence",
    "generate_masked_seq_str",
    "generate_masked_seqs_str",
    "get_masked_positions",
    "compute_diversity_metrics_str",
    "compute_diversity_metrics_int",
    "generate_masked_seq_arr",
    "append_csv_line",
    "get_mismatch_fraction_multiseqs",
    "read_fixed_positions",
    # MPNN workflow
    "run_mpnn",
    "get_mpnn_scores",
    "mpnn_masked_gen",
    "fill_mpnn",
    # ESM utilities
    "init_ESM",
    "get_embedding_single",
    "predict_location_from_emb",
    "predict_location_from_seq",
    "get_ESM_perplexity_one_pass",
    "predict_location_from_seqs",
    "get_ESM_perplexity_one_pass_multiseqs"
]
