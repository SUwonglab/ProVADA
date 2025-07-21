from dataclasses import dataclass
from typing import Any, Union, Optional
import numpy as np


# --------------------------------
# MADA parameters
# --------------------------------

@dataclass
class SamplerParams:
    """
    Holds configuration for running a masked‐sequence sampling round with ProteinMPNN.
    All fields are passed via the constructor and can be overridden as needed.
    """
    # Power-law exponent for annealing
    alpha: float = 3.0
    # Initial masking fraction
    init_mask_frac: float = 0.2
    # Minimum masking fraction
    min_mask_frac: float = 0.05
    # Optional: Schedule length for annealing
    T_schedule: Union[np.ndarray, list] = None

    # Top-k sequence to retain
    top_k_frac: float = 0.2
    # Lambda: penalty for residue mismatches between proposed and wt sequences
    mismatch_penalty: float = 0.2
    # Whether to use greedy top-k sampling
    greedy: bool = True 
    # Whether to gradually increase the mismatch penalty
    anneal_mismatch_penalty: bool = False 
    # Maximum mismatch penalty
    max_mismatch_penalty: float = 0.2
    # Optional: Schedule for annealing mismatch penalty
    mismatch_penalty_schedule: Union[np.ndarray, list] = None

    # Sampling temperature for MPNN
    mpnn_sample_temp: float = 0.5
    
