from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from provada.sampler.sampler_params import SamplerParams
from provada.sampler.tracker import TopProteinTracker

@dataclass
class SamplerResult:
    sampler_params: SamplerParams
    masked_sequences: List[str]
    filled_sequences: List[str]
    mpnn_scores: List[float]
    fitness_scores: List[float]
    classifier_probs: List[float]
    perplexities: List[float]
    trajectory: Optional[pd.DataFrame]
    top_tracker: TopProteinTracker