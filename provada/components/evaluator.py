"""
evaluator.py

Contains the Evaluator class, which is used to score sequences
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import os
import pathlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import pandas as pd
from provada.models.esm2 import ESM2Model
from provada.models.esm3 import ESM3Model

import io
import freesasa
from Bio.PDB import PDBParser

from provada.utils.env import device_manager, lock_via_lockfile
from provada.utils.multiprocess import get_pool, _worker_local

from provada.sequences.pairwise_metrics import (
    get_pairwise_metric,
    __all__ as pairwise_metric_functions,
)
from provada.utils.log import get_logger
from provada.sequences.io import hash_sequence
from provada.utils.registry import EVALUATOR_REGISTRY

logger = get_logger(__name__)


class Evaluator(ABC):

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = True,
    ):
        """
        Initializes the evaluator.

        Args:
            seed: The seed to use for the evaluator
            active_scores: The list of scores to activate for this evaluator. If
                not provided, all scores will be returned.
            use_cache: Whether to use the cache for the evaluator.
        """
        # Store the seed
        self.seed = seed

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

        # Run the initialization function for the evaluator
        init_fn = getattr(self, "worker_init")

        logger.debug(f"Running worker init function for {self.__class__.__name__}")
        get_pool().run_init(init_fn, self.__class__.__name__)

        # Get the ctx status
        ctx_status_list = get_pool().get_ctx_status()
        for ind, ctx_status in enumerate(ctx_status_list):
            # Check if worker has proper status structure
            if "success_list" not in ctx_status:
                raise Exception(
                    f"Worker {ind} has no initialization history - worker may have died during {self.__class__.__name__} initialization"
                )

            # Pull the latest status update
            success = ctx_status["success_list"][-1]
            retry_count = ctx_status["retry_count_list"][-1]
            message = ctx_status["message_list"][-1]

            if not success:
                raise Exception(
                    f"Error initializing {self.__class__.__name__} in worker {ind}: {message}"
                )
            elif retry_count > 0:
                logger.warning(
                    f"Was able to initialize {self.__class__.__name__} in worker {ind} after {retry_count} retries"
                )
            elif message != "":
                logger.debug(
                    f"Message from {self.__class__.__name__} in worker {ind}: {message}"
                )

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

    @staticmethod
    def worker_init():
        """
        Optional initialization for worker processes.
        Should return a dict of objects to be stored in the worker context.
        Example:
            return {"pyrosetta": pyrosetta, "pose_from_pdbstring": pose_from_pdbstring}
        """
        return {"success": True}

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
    ):
        # Do not allow the use of the cache for this evaluator
        if use_cache:
            logger.warning(
                "The cache is automatically disabled for the SequenceSimilarity evaluator since the reference sequence can change between evaluations"
            )
        self.only_consider_designable = only_consider_designable
        super().__init__(seed=seed, active_scores=active_scores, use_cache=False)

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
        return {
            "reference_sequence_or_sequences": base_variant.sequence,
            "fixed_position_indices": base_variant.fixed_position_indices,
            "sequence_length": base_variant.sequence_length,
        }

    def evaluate_sequences(
        self,
        sequences: List[str],
        reference_sequence_or_sequences: Union[str, List[str]],
        fixed_position_indices: List[int] = None,
        sequence_length: int = None,
        **kwargs,
    ) -> pd.DataFrame:

        # Prepare the sequences
        df = pd.DataFrame({"sequence": sequences})

        # Filter to only designable positions if enabled
        if (
            self.only_consider_designable
            and fixed_position_indices is not None
            and sequence_length is not None
        ):
            # Get designable positions (all positions not in fixed list)
            designable_positions = [
                i for i in range(sequence_length) if i not in fixed_position_indices
            ]

            # Filter sequences to only designable positions
            filtered_sequences = [
                "".join([seq[i] for i in designable_positions]) for seq in sequences
            ]

            # Filter reference sequence
            if isinstance(reference_sequence_or_sequences, str):
                filtered_reference = "".join(
                    [reference_sequence_or_sequences[i] for i in designable_positions]
                )
            else:
                # Handle list of reference sequences
                filtered_reference = [
                    "".join([ref[i] for i in designable_positions])
                    for ref in reference_sequence_or_sequences
                ]
        else:
            # Use full sequences
            filtered_sequences = sequences
            filtered_reference = reference_sequence_or_sequences

        df["reference_sequence"] = filtered_reference

        # Evaluate the sequences
        for score in self.active_scores:
            pairwise_metric_func = get_pairwise_metric(score)

            results = get_pool().process(
                pairwise_metric_func,
                [
                    (filtered_seq, ref)
                    for filtered_seq, ref in zip(filtered_sequences, df["reference_sequence"])
                ],
                show_progress=True,
            )

            df[score] = results

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

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = True,
    ):
        super().__init__(seed=seed, active_scores=active_scores, use_cache=use_cache)

        self.esm2_model = ESM2Model(model_name="esm2_t33_650M_UR50D")

        with open(os.path.join("inputs", "renin", "logreg_model.pkl"), "rb") as f:
            self.predictor = pickle.load(f)

    def evaluate_sequences(
        self, sequences: List[str], device: str = device_manager.get_device(), **kwargs
    ) -> pd.DataFrame:

        # Get embeddings
        outputs = self.esm2_model(
            sequences,
            batch_size=50,
            device=device,
            return_logits=False,
            progress_bar_msg="Running ESM2 localization predictor",
        )
        embeddings = outputs["mean_embeddings"]

        # Get predictions
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
        use_cache: bool = True,
    ):
        super().__init__(seed=seed, active_scores=active_scores, use_cache=use_cache)

        self.localization_evaluator = get_evaluator(
            "localization_predictor", seed=seed, use_cache=use_cache
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
        return {"reference_sequence_or_sequences": base_variant.sequence}

    def evaluate_sequences(
        self,
        sequences: List[str],
        reference_sequence_or_sequences: Union[str, List[str]],
        **kwargs,
    ) -> pd.DataFrame:

        # Get localization predictions
        localization_probabilities = self.localization_evaluator.evaluate_sequences(sequences)[
            "localization_prob"
        ].values

        normalized_hamming_distances = (
            self.normalized_hamming_distance_evaluator.evaluate_sequences(
                sequences=sequences,
                reference_sequence_or_sequences=reference_sequence_or_sequences,
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

    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
    ):
        super().__init__(seed=seed, active_scores=active_scores)

        self.esm3_model = ESM3Model()

    def evaluate_sequences(
        self,
        sequences: List[str],
        device: str = device_manager.get_device(),
        **kwargs,
    ) -> pd.DataFrame:

        # Predict structures
        structure_predictions = self.esm3_model.predict_structure(
            sequences=sequences, device=device
        )

        # Create DataFrame
        df = pd.DataFrame(structure_predictions)
        df = df.rename(columns={"avg_plddt": "esm3_avg_plddt", "ptm": "esm3_ptm"})

        # Get the sequence lengths
        seq_len = len(df["sequence"].iloc[0])

        # Use the pool
        logger.debug("Computing structure-based scores in parallel")
        results = get_pool().process(
            calculate_structure_scores,
            [(row["pdb_string"], self.active_scores, seq_len) for _, row in df.iterrows()],
        )
        scores_df = pd.DataFrame(results)

        # Combine with original DataFrame
        df = pd.concat([df, scores_df], axis=1)
        df.drop(columns=["pdb_string"], inplace=True)

        return df

    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "esm3_avg_plddt": {
                "larger_is_better": True,
                "min_value": 0,
                "max_value": 1,
            },
            "esm3_ptm": {"larger_is_better": True, "min_value": 0, "max_value": 1},
            "sap_score": {
                "larger_is_better": False,
                "min_value": None,  # 0 is minimum, but practically will be higher
                "max_value": None,
            },
            "sasa_score": {
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

    @staticmethod
    def worker_init():
        # Added locking for PyRosetta so processes don't conflict with each other
        lock_file_path = pathlib.Path(__file__).parent / "locks" / "StructureMetrics.lock"
        try:
            with lock_via_lockfile(lock_file_path, timeout=200):
                import pyrosetta

                pyrosetta.init("-mute all")
                from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
                from pyrosetta import get_fa_scorefxn
                from pyrosetta.rosetta.core.pack.guidance_scoreterms import (
                    sap as sap_terms,
                )

                return {
                    "pyrosetta": pyrosetta,
                    "pose_from_pdbstring": pose_from_pdbstring,
                    "get_fa_scorefxn": get_fa_scorefxn,
                    "sap_terms": sap_terms,
                    "success": True,
                }
        except Exception as e:
            return {"message": str(e), "success": False}


# ==============================
# Structure metrics
# ==============================
def pose_from_string(pdb_str: str, pyrosetta_module, pose_from_pdbstring_func):
    pose = pyrosetta_module.Pose()
    pose_from_pdbstring_func(pose, pdb_str)  # in-place fill
    pose.update_residue_neighbors()
    return pose


def calculate_structure_scores(
    pdb_string: str, active_scores: List[str], sequence_length: int
) -> dict:
    """
    Calculates the structure scores for a given PDB string.
    """
    ctx = getattr(_worker_local, "ctx", {})

    pose = pose_from_string(pdb_string, ctx["pyrosetta"], ctx["pose_from_pdbstring"])
    scores = {}
    if "sap_score" in active_scores:
        scores["sap_score"] = calculate_sap_score(pose, ctx["sap_terms"])
    if "sasa_score" in active_scores:
        scores["sasa_score"] = calculate_sasa_score(pdb_string) / sequence_length
    if "rosetta_energy_score" in active_scores:
        scores["rosetta_energy_score"] = (
            calculate_rosetta_energy_score(pose, ctx["get_fa_scorefxn"]) / sequence_length
        )
    return scores


def calculate_sap_score(pose, sap_terms) -> float:
    """
    Calculates the SAP score for a given PDB string. SAP (Spatial Aggregation
    Propensity) is a measure of the propensity of a protein to aggregate. The
    raw SAP score is a non-negative number where higher values indicate more
    aggregation.

    Args:
        pose: The PyRosetta pose to calculate the SAP score for
        sap_terms: The PyRosetta SAP terms to use

    Returns:
        The SAP score for the given PDB string
    """
    sm = sap_terms.SapScoreMetric()
    return float(sm.calculate(pose))


def calculate_sasa_score(pdb_string: str) -> float:
    """
    Calculates the SASA score for a given PDB string. SASA (Solvent Accessible
    Surface Area) is a measure of the total area of a protein that is accessible
    to a solvent probe. The raw SASA score is a non-negative number where higher
    values indicate more accessible surface area. For monomer stability, lower
    exposed hydrophic SASA is better.

    Args:
        pdb_string: The PDB string to calculate the SASA score for
    """
    # Parse the PDB string
    struct = PDBParser(QUIET=True).get_structure("X", io.StringIO(pdb_string))
    result, _atom_areas = freesasa.calcBioPDB(struct)

    return float(result.totalArea())


def calculate_rosetta_energy_score(pose, get_fa_scorefxn_func) -> float:
    """
    Calculates the Rosetta energy score for a given PDB string. The Rosetta total
    energy score is a negative number where lower values (more negative) indicate
    more stable structures.

    Args:
        pdb_string: The PDB string to calculate the Rosetta energy score for

    Returns:
        The rosetta energy score for the given PDB string
    """
    scorefxn = get_fa_scorefxn_func()  # standard full-atom score function
    total = scorefxn(pose)  # evaluates and caches energies on the pose
    return float(total)


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
