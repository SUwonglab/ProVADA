# base_sequence_info.py

"""
BaseSequenceInfo class to hold wild-type sequence information and compute initial scores.
This class is designed to encapsulate the configuration and initial scores for a sequence,
including classifier probabilities, ESM perplexity, and MPNN scores.
It is initialized with a sequence and computes the necessary scores upon instantiation.
"""


from dataclasses import dataclass, field
from typing import Any, Union
import numpy as np


from provada.utils.helpers import arr_to_aa
from provada.utils.esm_utils import (
    get_ESM_perplexity_one_pass,
    predict_location_from_seq,
)
from provada.utils.mpnn_utils import get_mpnn_scores


@dataclass
class BaseSequenceInfo:
    """
    Holds both configuration and initial scores for a sequence.
    On construction, computes:
      - base_classifier_prob
      - base_perplexity
      - base_MPNN_score
    """
    # Sequence input (AA string or integer array)
    base_seq: Union[str, np.ndarray]

    # Configuration fields with defaults
    classifier: Any            = None
    clf_name: str              = "logreg"
    target_label: str          = None
    ESM_model: Any             = None
    tokenizer: Any             = None
    device: str                = "cuda"
    penalty_perplexity: float  = 0.1
    penalty_MPNN: float        = 0.1
    input_pdb: str             = None
    save_path: str             = None
    protein_chain: str         = None
    hard_fixed_positions: list = field(default_factory=list) # empty list by default

    # computed initial values (do not pass)
    base_classifier_prob: float = field(init=False)
    base_perplexity: float      = field(init=False)
    base_MPNN_score: float      = field(init=False)


    def __post_init__(self):
        # Convert to string if needed
        seq_str = self.base_seq if isinstance(self.base_seq, str) else arr_to_aa(self.base_seq)

        # Classifier output
        loc_probs = predict_location_from_seq(
            seq=seq_str,
            clf_name=self.clf_name,
            classifier=self.classifier,
            ESM_model=self.ESM_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        if self.target_label == "cytosolic":
            idx = 0
        elif self.target_label == "extracellular":
            idx = 1
        else:
            raise ValueError("target_label must be 'cytosolic' or 'extracellular'")
        self.base_classifier_prob = loc_probs[idx]

        # ESM perplexity
        self.base_perplexity = get_ESM_perplexity_one_pass(
            seq=seq_str,
            model=self.ESM_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        # MPNN score
        self.base_MPNN_score = get_mpnn_scores(
            pdb=self.input_pdb,
            protein_chain=self.protein_chain,
        )
