"""
evaluator.py

Contains the Evaluator class, which is used to score sequences
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import numpy as np
import pandas as pd


from provada.paths import REPO_ROOT
from provada.utils.env import resolve_device

from provada.sequences.pairwise_metrics import (
    get_pairwise_metric,
    __all__ as pairwise_metric_functions,
)
from provada.utils.log import get_logger
from provada.sequences.io import hash_sequence
from provada.utils.registry import EVALUATOR_REGISTRY


logger = get_logger(__name__)

#: Localization classifier shipped with the renin example. Anchored to the
#: repository root so a run started from any working directory finds it.
DEFAULT_CLASSIFIER_PATH = REPO_ROOT / "inputs" / "renin" / "logreg_model.pkl"


class Evaluator(ABC):

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initializes the evaluator.

        Args:
            seed: The seed to use for the evaluator
            active_scores: The list of scores to activate for this evaluator. If
                not provided, all scores will be returned.
            use_cache: Whether to use the cache for the evaluator.
            device: The device string to forward to any model this evaluator
                calls. Passed through to proto-tools, whose DeviceManager owns
                placement; ``None`` leaves the tool's own default in place.
        """
        # Store the seed
        self.seed = seed

        # Forwarded to proto-tools tool configs; not used to move models here.
        self.device = device

        # Store the use cache flag
        self.use_cache = use_cache

        # Initialize the cache
        self.cache = None

        # Store the active scores (if provided)
        if active_scores is not None:
            # Ensure all active scores are valid
            for score in active_scores:
                if score not in self.available_scores().keys():
                    raise ValueError(
                        f"Invalid score: {score}, valid scores for {self.__class__.__name__} are: {self.available_scores()}"
                    )
        else:
            # Default to all available scores for this evaluator
            active_scores = list(self.available_scores().keys())

        self.active_scores = active_scores

        logger.info(
            f"Successfully initialized {self.__class__.__name__} with the following active scores: {self.active_scores}"
        )

    @abstractmethod
    def evaluate_sequences(self, sequences: List[str], **kwargs) -> pd.DataFrame:
        """
        Evaluates a list of sequences. Should be implemented by subclasses.
        """
        pass

    @classmethod
    @abstractmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the unique string key name for each score
        with a dictionary specifying the following attributes:
            - "larger_is_better": A boolean indicating whether larger values of the
                score are better.
            - "min_value": A float indicating the minimum value of the score. Specify
                None if the minimum value is not known.
            - "max_value": A float indicating the maximum value of the score. Specify
                None if the maximum value is not known.

        For the min and max values, you can also specify the string name of an attribute
        of base_variant, such as "sequence_length".
        """
        pass

    def update_cache(
        self,
        eval_df: Optional[pd.DataFrame] = None,
    ):
        """
        Updates the cache with entries in a new dataframe.
        """

        # If there are no evaluation outputs, cache is not updated
        if eval_df is None:
            return

        # Add the sequence hashes to the dataframe
        eval_df["sequence_hash"] = eval_df["sequence"].map(hash_sequence)

        # Drop non-active columns
        eval_df.drop(
            columns=[
                col
                for col in eval_df.columns
                if col not in self.active_scores + ["sequence_hash"]
            ],
            inplace=True,
        )

        # If the cache is None or the cache is turned off, overwrite cache with the dataframe
        if self.cache is None or not self.use_cache:
            # Set the cache to the new dataframe
            self.cache = eval_df
        else:
            # Otherwise, concatenate the new dataframe to the cache
            self.cache = pd.concat([self.cache, eval_df])

    def pre_eval(
        self,
        seq_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Pre-evaluate a list of sequences and return the dataframe with a flag
        indicating which sequences need to be evaluated.
        """

        # Set a flag to indicate which sequences need to be evaluated
        seq_df["new_sequence"] = True

        # If the cache is not None, set the flag to indicate which sequences are not in the cache
        if self.cache is not None and self.use_cache:
            seq_df["new_sequence"] = ~seq_df["sequence_hash"].isin(self.cache["sequence_hash"])

        # Return sequences
        return seq_df

    def post_eval(
        self,
        seq_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adds the evaluation outputs to the cache and returns the final dataframe.

        Args:
            seq_df: The input dataframe of sequences containing the "sequence"
                and "sequence_hash" columns in the original order
            eval_df: The dataframe of evaluation outputs containing the "sequence_hash"
                column in the original order

        Returns:
            The dataframe of scores
        """

        # Update the cache with the new entries
        self.update_cache(eval_df)

        # Merge the sequence list with the cache to extract the scores in the correct order
        scores_df = self.cache.merge(
            seq_df, on="sequence_hash", how="right", validate="one_to_many"
        )

        # Drop the "sequence_hash" column
        scores_df.drop(columns=["sequence_hash", "new_sequence"], inplace=True)

        # Return the scores
        return scores_df

    def evaluate(self, sequences: Union[List[str], pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Public wrapper that ensures pre_eval and post_eval are always called.
        This is the main entry point for evaluation.
        """

        # If a list is provided, convert to a dataframe
        seq_df = None
        if isinstance(sequences, list):
            seq_df = pd.DataFrame({"sequence": sequences})
        else:
            seq_df = sequences.copy()

        # Hash the incoming sequences
        seq_df["sequence_hash"] = seq_df["sequence"].map(hash_sequence)

        # Always call pre_eval first
        seq_df = self.pre_eval(seq_df)

        # Call the evaluation method on the unique new sequences
        eval_df = None
        if seq_df["new_sequence"].sum() > 0:
            try:
                eval_df = self.evaluate_sequences(
                    seq_df["sequence"][seq_df["new_sequence"]].unique().tolist(), **kwargs
                )
            except Exception as e:
                logger.error(
                    f"Error in evaluate_sequences function of {self.__class__.__name__}: {e}"
                )
                raise e
            if eval_df is None:
                raise ValueError(
                    f"evaluate_sequences method of {self.__class__.__name__} returned None"
                )
            
        # Always call post_eval after evaluation
        final_outputs = self.post_eval(seq_df, eval_df)

        return final_outputs

    def get_extra_kwargs(self, base_variant):
        """
        Returns a dictionary of extra keyword arguments needed for evaluation.
        Subclasses can override this method to specify additional kwargs required
        for their evaluate_sequences method.

        Args:
            base_variant: The base variant object that may contain information
                needed for evaluation (e.g., reference sequence, structure).

        Returns:
            A dictionary of extra keyword arguments.
        """
        return {}


# ==============================
# Evaluator instance cache
# ==============================

EVALUATOR_INSTANCE_CACHE: Dict[str, Evaluator] = {}


def get_evaluator(name: str, **kwargs) -> Evaluator:
    """
    Get an evaluator instance from the registry.
    """
    if name not in EVALUATOR_INSTANCE_CACHE:
        logger.debug(f"Creating new evaluator instance for {name}")
        # Create a new evaluator
        new_evaluator = EVALUATOR_REGISTRY.get_instance(name, **kwargs)
        EVALUATOR_INSTANCE_CACHE[name] = new_evaluator
    else:
        logger.debug(f"Evaluator cache hit for {name}")
        # Get the existing evaluator and update the active scores
        existing_evaluator = EVALUATOR_INSTANCE_CACHE[name]
        old_active_scores = existing_evaluator.active_scores
        new_active_scores = kwargs.get("active_scores", old_active_scores)
        final_active_scores = old_active_scores + [
            nas for nas in new_active_scores if nas not in old_active_scores
        ]

        # Update the active scores to include those included in either
        existing_evaluator.active_scores = final_active_scores

        # Use cache if it was True in either old or new
        existing_evaluator.use_cache = (
            kwargs.get("use_cache", True) or existing_evaluator.use_cache
        )

    # Return the evaluator instance
    return EVALUATOR_INSTANCE_CACHE[name]


def clear_all_evaluator_cache():
    """
    Fully clear all evaluator instances and their internal caches.
    """
    for ev in EVALUATOR_INSTANCE_CACHE.values():
        if hasattr(ev, "cache"):
            ev.cache = pd.DataFrame()  # or {} if using dict
    EVALUATOR_INSTANCE_CACHE.clear()


# ==============================
# Evaluator classes
# ==============================


@EVALUATOR_REGISTRY.register("dummy")
class Dummy(Evaluator):
    """
    A dummy evaluator for testing and development. Returns a score equal to the
    number of Methionine residues in the sequence.
    """

    def evaluate_sequences(self, sequences: List[str], **kwargs) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "sequence": sequence,
                "dummy_score": sequence.count("M"),
            }
            for sequence in sequences
        )

    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "dummy_score": {
                "larger_is_better": True,
                "min_value": 0,
                "max_value": "sequence_length",
            }
        }


@EVALUATOR_REGISTRY.register("sequence_similarity")
class SequenceSimilarity(Evaluator):
    """
    Returns similarity metrics between sequences
    """

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = False,
        only_consider_designable: bool = True,
        device: Optional[str] = None,
    ):
        # Do not allow the use of the cache for this evaluator
        if use_cache:
            logger.warning(
                "The cache is automatically disabled for the SequenceSimilarity evaluator since the reference sequence can change between evaluations"
            )
        self.only_consider_designable = only_consider_designable
        super().__init__(
            seed=seed, active_scores=active_scores, use_cache=False, device=device
        )

    @classmethod
    def available_scores(cls) -> List[str]:

        available_scores = {}
        for metric_name in pairwise_metric_functions:
            metric_func = get_pairwise_metric(metric_name)
            available_scores[metric_name] = {
                "larger_is_better": metric_func.more_similar_is_larger,
                "min_value": 0,
                "max_value": (
                    1 if metric_name != "levenshtein_distance" else "num_designable_positions"
                ),
            }

        return available_scores

    def get_extra_kwargs(self, base_variant):
        """
        SequenceSimilarity requires a reference sequence for comparison.
        """
        extra_kwargs = {
            "reference_sequence_or_sequences": base_variant.sequence,
            "fixed_position_indices": base_variant.fixed_position_indices,
            "sequence_length": base_variant.sequence_length,
        }

        # Check for single designable chain requirement if base variant is multichain
        if base_variant.sequence and '/' in base_variant.sequence:
             try:
                 extra_kwargs["designable_chain_index"] = base_variant.designable_chain_index
             except ValueError as e:
                 logger.warning(f"Could not determine designable chain index: {e}. Evaluator might fail or evaluate full sequence.")

        return extra_kwargs

    def _positions_to_compare(
        self,
        reference: str,
        fixed_position_indices: Optional[List[int]],
        sequence_length: Optional[int],
        designable_chain_index: Optional[int],
    ) -> Optional[List[int]]:
        """Which positions of the full sequence a metric should see.

        Both filters are applied in full-sequence coordinates — ``/`` separators
        included, the way ``base_variant.fixed_position_indices`` and
        ``sequence_length`` are measured — so neither has to rebase the other's
        indices.

        Args:
            reference: A reference sequence, used for its chain layout.
            fixed_position_indices: Positions held fixed during design.
            sequence_length: Length of the full sequence.
            designable_chain_index: Chain to restrict to, or None for all.

        Returns:
            Positions to keep, or None to compare the sequences whole.
        """
        length = sequence_length if sequence_length is not None else len(reference)
        keep = range(length)

        # Restrict to one chain. Its span is measured the way
        # base_variant.designable_chain_index measures it: a separator per chain.
        if designable_chain_index is not None:
            chains = reference.split("/")
            if designable_chain_index < len(chains):
                start = sum(len(chain) + 1 for chain in chains[:designable_chain_index])
                keep = range(start, start + len(chains[designable_chain_index]))

        positions = list(keep)

        # Drop the fixed positions, so only designed residues are compared.
        if self.only_consider_designable and fixed_position_indices is not None:
            fixed = set(fixed_position_indices)
            positions = [i for i in positions if i not in fixed]

        # A separator is not a residue, and matches itself in every pair, so it
        # would inflate every similarity score.
        positions = [i for i in positions if i < len(reference) and reference[i] != "/"]

        if len(positions) == len(reference):
            return None
        return positions

    def evaluate_sequences(
        self,
        sequences: List[str],
        reference_sequence_or_sequences: Union[str, List[str]],
        fixed_position_indices: Optional[List[int]] = None,
        sequence_length: Optional[int] = None,
        designable_chain_index: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:

        # Prepare the sequences
        df = pd.DataFrame({"sequence": sequences})

        # The chain layout and the fixed positions are the base variant's, so
        # one reference is enough to decide which positions to compare.
        first_reference = (
            reference_sequence_or_sequences
            if isinstance(reference_sequence_or_sequences, str)
            else reference_sequence_or_sequences[0]
        )
        positions = self._positions_to_compare(
            first_reference,
            fixed_position_indices,
            sequence_length,
            designable_chain_index,
        )

        def keep_positions(sequence: str) -> str:
            """Return only the positions being compared."""
            if positions is None:
                return sequence
            return "".join(sequence[i] for i in positions if i < len(sequence))

        filtered_sequences = [keep_positions(sequence) for sequence in sequences]
        if isinstance(reference_sequence_or_sequences, str):
            filtered_reference = keep_positions(reference_sequence_or_sequences)
        else:
            filtered_reference = [
                keep_positions(reference) for reference in reference_sequence_or_sequences
            ]

        df["reference_sequence"] = filtered_reference
        pairs = list(zip(filtered_sequences, df["reference_sequence"]))

        vectorised = vectorised_position_metrics(pairs, self.active_scores)

        for score in self.active_scores:
            if score in vectorised:
                df[score] = vectorised[score]
                continue
            pairwise_metric_func = get_pairwise_metric(score)
            df[score] = [
                pairwise_metric_func(sequence, reference) for sequence, reference in pairs
            ]

        df.drop(columns=["reference_sequence"], inplace=True)

        return df


@EVALUATOR_REGISTRY.register("localization_predictor")
class LocalizationPredictor(Evaluator):
    """
    Returns localization scores from predicted structures
    """

    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "localization_prob": {
                "larger_is_better": True,
                "min_value": 0,
                "max_value": 1, 
            }
        }

    #: ESM2 checkpoint whose mean embeddings the bundled classifier was fit on.
    #: Changing it invalidates the classifier, so it is not a config knob.
    MODEL_CHECKPOINT = "esm2_t33_650M_UR50D"

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = True,
        device: Optional[str] = None,
        classifier_path: Optional[str] = None,
        batch_size: int = 50,
    ):
        """
        Initializes the localization predictor.

        Args:
            seed: The seed to use for the evaluator.
            active_scores: Which of this evaluator's scores to return.
            use_cache: Whether to cache results by sequence hash.
            device: Device string forwarded to the ESM2 tool.
            classifier_path: Pickled scikit-learn classifier mapping an ESM2
                mean embedding to a localization probability. Defaults to
                ``DEFAULT_CLASSIFIER_PATH``, resolved relative to the repository
                root rather than the working directory.
            batch_size: Sequences per ESM2 forward pass.
        """
        super().__init__(
            seed=seed,
            active_scores=active_scores,
            use_cache=use_cache,
            device=device,
        )

        self.batch_size = batch_size
        self.classifier_path = Path(classifier_path or DEFAULT_CLASSIFIER_PATH)
        if not self.classifier_path.is_file():
            raise FileNotFoundError(
                f"No localization classifier at {self.classifier_path}. Pass "
                "classifier_path to point at one trained for your target."
            )
        with open(self.classifier_path, "rb") as f:
            self.predictor = pickle.load(f)

    def get_extra_kwargs(self, base_variant):
        """
        LocalizationPredictor may require extracting the designable chain.
        """
        extra_kwargs = {}
        # Check for single designable chain requirement if base variant is multichain
        if base_variant.sequence and '/' in base_variant.sequence:
             try:
                 extra_kwargs["designable_chain_index"] = base_variant.designable_chain_index
             except ValueError:
                 pass
        return extra_kwargs

    def evaluate_sequences(
        self,
        sequences: List[str],
        designable_chain_index: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Score sequences by localization probability.

        Embeddings come from proto-tools' ``esm2-embedding`` tool, which runs
        ESM2 in its own environment; only the classifier runs in this process.

        Args:
            sequences: Sequences to score. A multi-chain sequence is joined by
                ``/``, and the returned frame keys on the sequence as given.
            designable_chain_index: Which ``/``-separated chain to embed. The
                classifier was fitted on single-chain embeddings, so for a
                multi-chain variant only the designable chain is passed to the
                model. Supplied by ``get_extra_kwargs``.
        """
        from proto_tools.tools import (
            ESM2EmbeddingsConfig,
            ESM2EmbeddingsInput,
            run_esm2_embeddings,
        )

        sequences_to_eval = sequences
        if designable_chain_index is not None:

            def extract_chain(seq_str, idx):
                parts = seq_str.split('/')
                if idx >= len(parts):
                    return seq_str
                return parts[idx]

            sequences_to_eval = [
                extract_chain(s, designable_chain_index) for s in sequences
            ]

        config_kwargs = {
            "model_checkpoint": self.MODEL_CHECKPOINT,
            "batch_size": self.batch_size,
            "return_logits": False,
            "seed": self.seed,
        }
        config_kwargs["device"] = resolve_device(self.device)

        outputs = run_esm2_embeddings(
            ESM2EmbeddingsInput(sequences=sequences_to_eval),
            ESM2EmbeddingsConfig(**config_kwargs),
        )

        embeddings = [result.mean_embedding for result in outputs.results]
        predictions = self.predictor.predict_proba(embeddings)
        return pd.DataFrame({"sequence": sequences, "localization_prob": predictions[:, 1]})


@EVALUATOR_REGISTRY.register("discounted_hamming_distance")
class DiscountedHammingDistance(Evaluator):
    """
    Returns discounted hamming distance to reference sequence (normalized by sequence length)
    and multiply by localization probability.

    NOTE: This is an example of how we recommend combining evalutors into a
    single evaluator.
    """

    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "discounted_hamming_distance": {
                "larger_is_better": False,
                "min_value": 0,
                "max_value": 1,
            }
        }

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__(
            seed=seed,
            active_scores=active_scores,
            use_cache=use_cache,
            device=device,
        )

        self.localization_evaluator = get_evaluator(
            "localization_predictor", seed=seed, use_cache=use_cache, device=device
        )
        self.normalized_hamming_distance_evaluator = get_evaluator(
            "sequence_similarity",
            seed=seed,
            active_scores=["normalized_hamming_distance"],
            use_cache=use_cache,
        )

    def get_extra_kwargs(self, base_variant):
        """
        DiscountedHammingDistance requires a reference sequence for comparison.
        """
        extra_kwargs = {
            "reference_sequence_or_sequences": base_variant.sequence,
        }
        
        # Check for single designable chain requirement if base variant is multichain
        if base_variant.sequence and '/' in base_variant.sequence:
             try:
                 extra_kwargs["designable_chain_index"] = base_variant.designable_chain_index
             except ValueError:
                 pass
                 
        return extra_kwargs

    def evaluate_sequences(
        self,
        sequences: List[str],
        reference_sequence_or_sequences: Union[str, List[str]],
        designable_chain_index: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:

        # Filter sequences for localization_predictor if designable chain index is provided
        sequences_for_loc = sequences
        if designable_chain_index is not None:
             # Extract the designable chain
            def extract_chain(seq_str, idx):
                parts = seq_str.split('/')
                if idx >= len(parts):
                    return seq_str
                return parts[idx]

            sequences_for_loc = [extract_chain(s, designable_chain_index) for s in sequences]

        # Get localization predictions
        localization_probabilities = self.localization_evaluator.evaluate_sequences(sequences_for_loc)[
            "localization_prob"
        ].values

        normalized_hamming_distances = (
            self.normalized_hamming_distance_evaluator.evaluate_sequences(
                sequences=sequences,
                reference_sequence_or_sequences=reference_sequence_or_sequences,
                designable_chain_index=designable_chain_index,
                **kwargs,
            )["normalized_hamming_distance"].values
        )

        # Combine
        discounted_hamming_distances = (
            normalized_hamming_distances * localization_probabilities
        )

        return pd.DataFrame(
            {
                "sequence": sequences,
                "discounted_hamming_distance": discounted_hamming_distances,
            }
        )


@EVALUATOR_REGISTRY.register("structure")
class StructureMetrics(Evaluator):
    """
    Returns protein quality scores from predicted structures
    """

    def evaluate_sequences(
        self,
        sequences: List[str],
        device: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:

        pdb_strings, plddts, ptms = dispatch_structure_predictions(
            sequences, device if device is not None else self.device
        )

        df = pd.DataFrame(
            {
                "sequence": sequences,
                "esmfold2_avg_plddt": plddts,
                "esmfold2_ptm": ptms,
            }
        )

        # Scores are reported per residue, so a longer design is not penalised
        # for its length alone.
        seq_len = len(sequences[0])

        if "sap_score" in self.active_scores:
            df["sap_score"] = dispatch_sap_scores(pdb_strings)

        if "rosetta_energy_score" in self.active_scores:
            df["rosetta_energy_score"] = [
                energy / seq_len for energy in dispatch_rosetta_energy_scores(pdb_strings)
            ]

        return df

    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "esmfold2_avg_plddt": {
                "larger_is_better": True,
                "min_value": 0,
                "max_value": 1,
            },
            "esmfold2_ptm": {"larger_is_better": True, "min_value": 0, "max_value": 1},
            "sap_score": {
                "larger_is_better": False,
                "min_value": None,  # 0 is minimum, but practically will be higher
                "max_value": None,
            },
            "rosetta_energy_score": {
                "larger_is_better": False,
                "min_value": None,
                "max_value": None,  # 0 is max, but practically will be more negative
            },
        }


# ==============================
# Structure metrics
# ==============================
def dispatch_sap_scores(pdb_strings: List[str]) -> List[float]:
    """
    Calculates the SAP score for each PDB string. SAP (Spatial Aggregation
    Propensity) is a measure of the propensity of a protein to aggregate. The
    raw SAP score is a non-negative number where higher values indicate more
    aggregation.

    This is ProVADA's measure of exposed hydrophobicity. It weights each
    residue's sidechain exposure by its hydrophobicity and sums over a spatial
    neighbourhood, so it scores hydrophobic *patches* rather than total apolar
    area. Two designs can expose the same area and aggregate very differently
    depending on whether that area is clustered, which a scalar area cannot
    distinguish. ``pyrosetta-sasa`` reports hydrophobic area if the raw
    quantity is ever wanted alongside this.

    Args:
        pdb_strings: The PDB strings to score

    Returns:
        One SAP score per input, in input order
    """
    from proto_tools.tools import (
        PyRosettaSAPConfig,
        PyRosettaSAPInput,
        run_pyrosetta_sap,
    )

    output = run_pyrosetta_sap(
        PyRosettaSAPInput(inputs=list(pdb_strings)), PyRosettaSAPConfig()
    )
    return [float(result.sap_score) for result in output.results]


#: Scores that are a straight position-by-position comparison, so a whole batch
#: of equal-length pairs is one array comparison rather than one call each.
POSITION_WISE_METRICS = frozenset({"sequence_identity", "normalized_hamming_distance"})


def vectorised_position_metrics(
    pairs: List[Tuple[str, str]], active_scores: List[str]
) -> Dict[str, List[float]]:
    """
    Computes the position-wise similarity scores for a whole batch at once.

    ``sequence_identity`` and ``normalized_hamming_distance`` both reduce to a
    count of matching positions when the two sequences are the same length,
    which is the case for a population of designs against one reference. That
    makes the batch a single NumPy comparison instead of one Python call per
    pair.

    Each score is derived from the match count independently. Deriving the
    Hamming distance as ``1 - identity`` instead would disagree with the
    per-pair implementation in the last bit for most lengths, because the two
    expressions round differently.

    Args:
        pairs: ``(sequence, reference)`` pairs, already filtered to the
            positions being compared
        active_scores: Scores the evaluator was asked for

    Returns:
        One list of scores per position-wise metric that was requested, empty
        when the batch is ragged or nothing applies
    """
    wanted = POSITION_WISE_METRICS & set(active_scores)
    if not wanted or not pairs:
        return {}

    # The comparison needs one common width; anything else falls back to the
    # per-pair functions, which align before scoring.
    lengths = {len(sequence) for pair in pairs for sequence in pair}
    if len(lengths) != 1:
        return {}
    (length,) = lengths
    if length == 0:
        return {}

    sequences = np.frombuffer("".join(pair[0] for pair in pairs).encode(), dtype=np.uint8)
    references = np.frombuffer("".join(pair[1] for pair in pairs).encode(), dtype=np.uint8)
    if sequences.size != len(pairs) * length:
        # A non-ASCII residue code makes the byte count disagree with the
        # character count, so the reshape below would silently misalign.
        return {}
    matches = (
        (sequences.reshape(len(pairs), length) == references.reshape(len(pairs), length))
        .sum(axis=1)
        .tolist()
    )

    results = {}
    if "sequence_identity" in wanted:
        results["sequence_identity"] = [count / length for count in matches]
    if "normalized_hamming_distance" in wanted:
        results["normalized_hamming_distance"] = [
            (length - count) / length for count in matches
        ]
    return results


def dispatch_structure_predictions(
    sequences: List[str], device: Optional[str]
) -> Tuple[List[str], List[float], List[float]]:
    """
    Predicts a structure for each sequence with ESMFold2.

    Args:
        sequences: Sequences to fold, chains split by '/'
        device: Device to fold on, or None for the tool's default

    Returns:
        PDB content, mean pLDDT and pTM, one of each per input, in input order
    """
    from proto_tools.entities.complex import Chain, Complex
    from proto_tools.tools import ESMFold2Config, ESMFold2Input, run_esmfold2

    complexes = [
        Complex(
            chains=[
                Chain(sequence=chain, entity_type="protein")
                for chain in sequence.split("/")
            ]
        )
        for sequence in sequences
    ]

    config_kwargs = {"device": resolve_device(device)}
    output = run_esmfold2(
        ESMFold2Input(complexes=complexes), ESMFold2Config(**config_kwargs)
    )

    pdb_strings, plddts, ptms = [], [], []
    for structure in output.structures:
        pdb_content, _chain_map = structure.to_pdb_with_chain_mapping()
        pdb_strings.append(pdb_content)
        plddts.append(float(structure.metrics.plddt))
        ptms.append(float(structure.metrics.ptm))
    return pdb_strings, plddts, ptms


def dispatch_rosetta_energy_scores(pdb_strings: List[str]) -> List[float]:
    """
    Calculates the total Rosetta energy for each PDB string. The score is a
    negative number where lower values (more negative) indicate more stable
    structures.

    Scored with ``ref2015``, the score function ``get_fa_scorefxn()`` returns by
    default.

    Args:
        pdb_strings: The PDB strings to score

    Returns:
        One total energy, in REU, per input, in input order
    """
    from proto_tools.tools import (
        PyRosettaEnergyConfig,
        PyRosettaEnergyInput,
        run_pyrosetta_energy,
    )

    output = run_pyrosetta_energy(
        PyRosettaEnergyInput(inputs=list(pdb_strings)),
        PyRosettaEnergyConfig(scorefxn="ref2015"),
    )
    return [float(result.total_energy) for result in output.results]


# ==============================
# Helper functions
# ==============================
def get_evaluators_from_score_keys(score_keys: List[str]) -> List[Evaluator]:
    """
    Returns a list of evaluators that calculate the given scores.
    """

    # Ensure all score keys are valid
    valid_score_keys = list_all_score_keys()
    for score_key in score_keys:
        if score_key not in valid_score_keys:
            raise ValueError(
                f"Invalid score key: {score_key}, valid score keys are: {valid_score_keys}"
            )

    # Collect all available evaluators
    evaluators = EVALUATOR_REGISTRY.list_available_classes()

    # Filter the evaluators to only include those that calculate the given scores
    evaluators = [
        evaluator
        for evaluator in evaluators
        if set(evaluator.available_scores()).intersection(score_keys)
    ]

    return evaluators


def list_all_score_keys() -> List[str]:
    """
    Lists all the unique score keys that are available across all available evaluators.
    """
    evaluator_classes = EVALUATOR_REGISTRY.list_available_classes()
    return list(
        set(
            [
                score_key
                for evaluator in evaluator_classes
                for score_key in evaluator.available_scores()
            ]
        )
    )


# ==============================
# immunogenicity evaluator
# ==============================

@EVALUATOR_REGISTRY.register("Immunogenicity")
class ImmunogenicityMetrics(Evaluator):
    """
    Returns immunogenicity metrics from netMHCpan predictions
    """
    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "immunogenicity_score": {
                "larger_is_better": False,
                "min_value": 0,
                "max_value": 100,
            },
            "allele_averaged_positive_pep_ratio": {
                "larger_is_better": False,
                "min_value": 0,
                "max_value": 1,
            }
        }

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        device: Optional[str] = None,
    ):
        # Accepted and stored for interface symmetry; netMHCpan is a CPU binary,
        # so this evaluator never dispatches to a device.
        super().__init__(seed=seed, active_scores=active_scores, device=device)


    def evaluate_sequences(
        self, sequences: List[str], **kwargs
    ) -> pd.DataFrame:

        mhc_alleles = kwargs.get("mhc_alleles", None)

        self_peptide_db = kwargs.get("self_peptide_db", None)

        if mhc_alleles is None or len(mhc_alleles) == 0:
            raise ValueError("MHC alleles list cannot be empty for immunogenicity scoring.")

        scans = dispatch_epitope_scans(sequences, mhc_alleles)

        df = pd.DataFrame({"sequence": sequences})
        rows = [
            score_epitope_scan(scan, len(mhc_alleles), self_peptide_db) for scan in scans
        ]
        scores_df = pd.DataFrame(rows)

        columns_to_keep = [
            score
            for score in ("immunogenicity_score", "allele_averaged_positive_pep_ratio")
            if score in self.active_scores
        ]

        return pd.concat([df, scores_df[columns_to_keep]], axis=1)

    def get_extra_kwargs(self, base_variant):
        """
        Immunogenicity metrics requires additional MHC alleles and optional self peptide database.
        """
        return {
            "mhc_alleles": base_variant.mhc_alleles,
            "self_peptide_db": base_variant.self_peptide_db if hasattr(base_variant, "self_peptide_db") else None,
        }

# scoring function
#: Peptide length ProVADA scans. Class I epitopes are 8-11mers; this scores 9mers
#: only, which is what the published runs used.
PEPTIDE_LENGTH = 9


def dispatch_epitope_scans(sequences: List[str], alleles: List[str]) -> List[Any]:
    """
    Predicts MHC class I binding for every window of every sequence.

    Args:
        sequences: Protein sequences to scan
        alleles: MHC alleles to predict against, in any accepted spelling

    Returns:
        One scan per input sequence, in input order, each carrying every
        peptide-by-allele prediction
    """
    from proto_tools.tools import (
        NetMHCpanProteinConfig,
        NetMHCpanProteinInput,
        run_netmhcpan_protein,
    )

    output = run_netmhcpan_protein(
        NetMHCpanProteinInput(sequences=list(sequences)),
        NetMHCpanProteinConfig(
            alleles=list(alleles),
            peptide_lengths=[PEPTIDE_LENGTH],
            # Scoring reads the binding-affinity rank, which -BA adds alongside
            # the eluted-ligand columns rather than replacing them.
            include_binding_affinity=True,
        ),
    )
    return output.results


def score_epitope_scan(
    scan: Any,
    num_alleles: int,
    self_peptide_db: Optional[set] = None,
    threshold_rank: float = 10.0,
    k: float = 0.5,
) -> Dict[str, float]:
    """
    Reduces one protein's predictions to ProVADA's two immunogenicity scores.

    Each peptide-allele pair contributes a logistic score in the binding-affinity
    rank. A peptide already present in the self-peptide database is dropped from
    both aggregates, since a presented self peptide is tolerated.

    Args:
        scan: One protein's predictions, from ``dispatch_epitope_scans``
        num_alleles: Size of the allele panel, which both scores average over
        self_peptide_db: Peptides to treat as self, or None to count every peptide
        threshold_rank: Rank at the logistic's midpoint, and the positive cutoff
        k: Steepness of the logistic

    Returns:
        ``immunogenicity_score`` and ``allele_averaged_positive_pep_ratio``
    """
    scores = []
    positive_peptide_count = 0

    for binding in scan.bindings:
        rank = binding.rank_ba
        score = logistic_map_rank_to_score(rank, k=k, x0=threshold_rank)
        # Only check the self database for peptides that would otherwise score.
        if (self_peptide_db is not None) and (score > 0.05):
            if find_endogenous_peptide(binding.peptide, self_peptide_db):
                continue
        scores.append(score)
        if rank <= threshold_rank:
            positive_peptide_count += 1

    return {
        "immunogenicity_score": sum(scores) / num_alleles,
        "allele_averaged_positive_pep_ratio": (
            positive_peptide_count / num_alleles / scan.sequence_length
        ),
    }


# netmhcpan helpers
# score mapping, map rank (0-100) to score (1-0)
def logistic_map_rank_to_score(rank: float, k: float = 0.5, x0: float = 10.0) -> float:
    """
    Map a rank (0-100) to a score (0-1) using a logistic function.
    k: steepness of the curve
    x0: the rank value of the sigmoid's midpoint
    """
    import math
    if rank < 0 or rank > 100:
        raise ValueError("Rank must be between 0 and 100.")
    score =  1- 1 / (1 + math.exp(-k * (rank - x0)))
    return score





def find_endogenous_peptide(peptide, endogenous_peptides_set: set) -> pd.DataFrame:
    """
    find if a peptide is in the endogenous peptides set
    """
    return peptide in endogenous_peptides_set



# ==============================
# Generator Score Evaluators
# ==============================

class GeneratorEvaluator(Evaluator):
    """
    Base class for evaluators that extract scores from the input dataframe
    provided by the generator.
    """
    def evaluate(self, sequences: Union[List[str], pd.DataFrame], **kwargs) -> pd.DataFrame:
        if isinstance(sequences, list):
            logger.warning(f"{self.__class__.__name__} received list of sequences. Cannot extract generator scores.")
            return pd.DataFrame({score: [float('nan')] * len(sequences) for score in self.active_scores})

        # Return a dataframe with just the requested scores
        result = pd.DataFrame(index=sequences.index)
        for score in self.active_scores:
            if score in sequences.columns:
                result[score] = sequences[score]
            else:
                result[score] = float('nan')
        return result
        
    def evaluate_sequences(self, sequences: List[str], **kwargs) -> pd.DataFrame:
        raise NotImplementedError("GeneratorEvaluator should not call evaluate_sequences")


@EVALUATOR_REGISTRY.register("caliby_energy_score")
class CalibyGeneratorEvaluator(GeneratorEvaluator):
    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "caliby_energy_score": {
                "larger_is_better": False, 
                "min_value": -1000,
                "max_value": 0
            }
        }

@EVALUATOR_REGISTRY.register("mpnn_score")
class MPNNGeneratorEvaluator(GeneratorEvaluator):
    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "mpnn_score": {
                "larger_is_better": False, 
                "min_value": 0,
                "max_value": 3
            }
        }
