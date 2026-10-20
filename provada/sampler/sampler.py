"""
sampler.py

Implements a class for Samplers, the primary construct for the ProVADA codebase.
Samplers are responsible for calling other components and directing the flow of
the search space.
"""

import os
import numpy as np
from pathlib import Path
from math import ceil
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from contextlib import ExitStack, nullcontext

from typing import Dict, List, Any, Optional, Tuple, Literal
from provada.base_variant import BaseVariant
from provada.sequences.mask import mask_p
from provada.utils.log import get_logger
from provada.utils.setup import seed_everything
from provada.utils.registry import SAMPLER_REGISTRY
from provada.utils.cache import shutdown_cache
from provada.sampler.tracking import TrackingMixin
from provada.sampler.startup import StartupMixin
from provada.sampler.reference_ranges import ReferenceRangesMixin
import wandb
import time




logger = get_logger(__name__)


def get_sampler_class(name: str) -> "Sampler":
    """
    Returns a sampler class of the given name.
    """
    return SAMPLER_REGISTRY.get_class(name=name)


class Sampler(ABC, TrackingMixin, StartupMixin, ReferenceRangesMixin):
    """
    Base sampler class. Implements the primary construct for the ProVADA codebase.
    """

    def __init__(
        self,
        base_variant: BaseVariant,
        num_iters: int,
        population_size: int,
        top_k_fraction: float,
        score_weights: Dict[str, float],
        generator_type: str = "random",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        masking_schedule_config: Optional[Dict[str, Any]] = None,
        temperature_schedule_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        reference_distribution_file: Optional[str] = None,
        wandb_run: Optional[wandb.Run] = None,
        config: Optional[Dict[str, Any]] = None,
        mhc_alleles: Optional[str] = None,
        greedy: bool = True,
        normalize: bool = True,
        resampling_method: Literal["systematic", "multinomial"] = "systematic",
    ):
        """
        Initialize the sampler

        Args:
            base_variant: The base variant. (Starting point of directed evolution)
            num_iters: The number of iterations to run the sampler for.
            population_size: The number of sequences to maintain in each iteration.
            top_k_fraction: The fraction of sequences to select from the generated variants using
                the select_variants method.
            score_weights: A dictionary containing a mapping of score keys to
                various weights.
            generator_type: The type of generator to use.
            generation_kwargs: The kwargs to pass to the generation function
            masking_schedule_config: The configuration for the masking schedule.
            temperature_schedule_config: The configuration for the temperature schedule.
            seed: Seed to use for various components of the sampler.
            run_name: The name of the run.
            output_dir: The directory to save the results to. If not provided, defaults to "results".
            reference_distribution_file: The file to load the reference distribution from. If not provided, defaults to None.
            wandb_run: The wandb run to use. If not provided, defaults to None.
            config: Optional config to save to the output directory. NOTE: The
                config will not be used for anything other than saving to the output directory.
            mhc_alleles: Comma-separated string of MHC alleles for immunogenicity scoring. If not provided, defaults to None.
            greedy: For mada, enable top-k selection and weighted population reconstruction (default True).
            normalize: Scale metrics with fixed bounds and return fitness in [0,1].
                False combines oriented raw metrics instead. SCORE_delta records
                improvement relative to the reference in either mode.
            resampling_method: For non-greedy SMC, use systematic (default) or
                multinomial resampling of the full weighted population.

        """
        # Set input attributes
        self.base_variant = base_variant
        self.num_iters = num_iters
        self.population_size = population_size
        self.top_k_fraction = top_k_fraction
        self.seed = seed
        self.greedy = greedy
        self.normalize = normalize
        if resampling_method not in ("systematic", "multinomial"):
            raise ValueError("resampling_method must be 'systematic' or 'multinomial'")
        self.resampling_method = resampling_method
        self.active_scores_dict = {
            score: {"weight": weight} for score, weight in score_weights.items()
        }
        self.wandb_run = wandb_run
        self.reference_distribution_file = reference_distribution_file
        self._fitness_weights()
        self._generator_score_cache = {}

        # Keep proto-tools model workers warm for the whole sampler lifetime.
        # Entered before the first evaluator runs, because initialize_evaluators
        # scores the base variant and _determine_min_max_ranges can push
        # thousands of reference variants through the generator before iteration
        # one. Startup failures and run() both release the owned context.
        self._compute_stack = ExitStack()
        self._resources_released = False
        self._ended = False
        try:
            if self.base_variant.num_designable_positions < 1:
                raise ValueError("Sampling requires at least one designable residue; all residues are fixed")
            self._compute_stack.enter_context(self._compute_context())

            # Seed everything
            seed_everything(self.seed)

            # Generator set up
            # initialize generator BEFORE evaluators so that generator-based scores can be computed
            self.generation_kwargs = generation_kwargs
            self.generator = None
            self.initialize_generator(generator_type, generation_kwargs)

            # Evaluator set up
            self.evaluators = []
            self.initialize_evaluators()

            # Masking schedule set up
            self.masking_schedule = None
            self.initialize_masking_schedule(masking_schedule_config)

            # Temperature schedule set up
            self.temperature_schedule = None
            self.initialize_temperature_schedule(temperature_schedule_config)

            # Set up run name if not provided
            if run_name is None:
                run_name = f"sampler_run_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"

            self.run_name = run_name

            # Set up output directory
            if output_dir is None:
                output_dir = Path("results")

            self.output_dir = Path(output_dir) / run_name

            os.makedirs(self.output_dir, exist_ok=True)

            # If config is provided, save it to the output directory
            if config is not None:
                config.save_yaml(self.output_dir / f"{run_name}_config.yaml")

            self.iteration = 0

            # Tracking
            self.observed_value_stats = {
                score_name: {"min": np.inf, "max": -np.inf}
                for score_name in self.active_scores_dict.keys()
            }
            self.observed_value_stats["SCORE"] = {"min": np.inf, "max": -np.inf}

            self.run_stats = {
                "num_unique_sequences": 0,
                "total_sequences_evaluated": 0,
            }

            # Collection of sequence hashes of sequences we have seen
            self.sequence_hashes = set()

            if self.wandb_run is not None:
                # Log the base variant scores
                self.wandb_run.summary.update(
                    {"base_variant_scores": self.base_variant.score_dict}
                )

            logger.info(
                f"Sampler {self.__class__.__name__} run ({self.run_name}) initialized. Results will be cached to {self.output_dir}"
            )
        except BaseException as error:
            # Construction has not started CSV writers or produced complete
            # outputs. Release only the resources acquired during startup.
            cleanup_errors = self._release_resources(flush_cache=False)
            self._finish_tracking(status="error", errors=cleanup_errors)
            self._ended = True
            for label, cleanup_error in cleanup_errors:
                error.add_note(f"Cleanup failed during {label}: {cleanup_error}")
            raise

    # ===== Helper Functions =====
    def mask_sequences(self, iteration_df: pd.DataFrame):
        """
        Masks a random fraction of each sequence's designable positions.

        The fraction comes from the masking schedule. ProteinMPNN shares one
        mask across repeated copies of an ancestor within this iteration, so
        all their completions can be batched. Other generators mask each row
        independently. Every particle retains its own parent metadata.
        """
        fraction = self.masking_schedule(iteration=self.iteration)
        fixed = self.base_variant.fixed_position_indices
        if getattr(self.generator, "batch_shared_prompts", False):
            masked_by_ancestor = {}
            for sequence in iteration_df["sequence"]:
                if sequence not in masked_by_ancestor:
                    masked_by_ancestor[sequence] = mask_p(sequence, p=fraction, fixed_indices=fixed)
            iteration_df["prompt"] = iteration_df["sequence"].map(masked_by_ancestor)
            return iteration_df

        iteration_df["prompt"] = [
            mask_p(
                sequence,
                p=fraction,
                fixed_indices=fixed,
            )
            for sequence in iteration_df["sequence"]
        ]
        return iteration_df

    def generate_variants(self, iteration_df: pd.DataFrame, **generation_overrides):
        """
        Generates a series of sequences using the generator.
        """
        # Copy so the per-generator extras below don't accumulate across iterations.
        generate_kwargs = dict(self.generation_kwargs)

        # Get any extra kwargs needed by this generator
        extra_kwargs = self.generator.get_extra_kwargs(self.base_variant)
        generate_kwargs.update(extra_kwargs)
        generate_kwargs.update(generation_overrides)

        return self.generator.generate(
            iteration_df,
            **generate_kwargs,
        )

    @staticmethod
    def _finite_scores(values, name):
        """Reject missing, nonnumeric, or nonfinite required metric values."""
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Active score '{name}' must be numeric and finite") from error
        if not np.isfinite(array).all():
            raise ValueError(f"Active score '{name}' contains missing or nonfinite values")
        return array

    def _fitness_weights(self):
        weights = np.asarray([info["weight"] for info in self.active_scores_dict.values()], dtype=float)
        if len(weights) == 0 or not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
            raise ValueError("Score weights must be finite, nonnegative, and have a positive total")
        weights = weights / weights.max()
        return weights / weights.sum()

    def _fitness_component(self, values, name, info):
        values = self._finite_scores(values, name)
        if not getattr(self, "normalize", True):
            return values if info["larger_is_better"] else -values
        lower, upper = info["min_value"], info["max_value"]
        try:
            valid = np.isfinite([lower, upper]).all() and upper > lower
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"Active score '{name}' requires finite bounds with max_value > min_value")
        values = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
        return values if info["larger_is_better"] else 1.0 - values

    def _canonical_generator_scores(self, iteration_df, score_subset=None):
        """Use fixed-seed sequence scores, never proposal-time MPNN scores.

        The cache is specific to the generator, backbone, chain order, checkpoint
        and scoring seed. Each unseen unique sequence is scored once in a batch.
        """
        needed = set(self.active_scores_dict if score_subset is None else score_subset)
        if "mpnn_score" not in needed:
            return iteration_df
        frame = iteration_df.copy()
        if "sequence" not in frame or not frame["sequence"].map(lambda value: isinstance(value, str)).all():
            raise ValueError("Canonical MPNN scoring requires a sequence string for every row")
        structure = self.base_variant.structure
        if structure is None:
            raise ValueError("Canonical MPNN scoring requires a backbone structure")
        generator = self.generator
        backbone = getattr(structure, "structure_pdb", str(structure))
        chains = tuple(structure.get_chain_ids()) if hasattr(structure, "get_chain_ids") else ()
        context = (id(generator), getattr(generator, "seed", None),
                   getattr(generator, "_TOOLKIT", None), getattr(generator, "_WEIGHTS", None),
                   getattr(generator, "device", None), backbone, chains)
        cache = getattr(self, "_generator_score_cache", None)
        if cache is None:
            cache = self._generator_score_cache = {}
        unique = frame["sequence"].drop_duplicates().tolist()
        unseen = [sequence for sequence in unique if (context, sequence) not in cache]
        if unseen:
            scored = generator.score(
                sequences=unseen, **generator.get_extra_kwargs(self.base_variant)
            )
            if len(scored) != len(unseen) or "mpnn_score" not in scored:
                raise ValueError("Canonical MPNN scorer must return one mpnn_score per sequence")
            values = self._finite_scores(scored["mpnn_score"], "mpnn_score")
            cache.update({(context, sequence): float(value) for sequence, value in zip(unseen, values)})
        if "mpnn_score" in frame:
            frame["generation_mpnn_score"] = frame["mpnn_score"]
        frame["mpnn_score"] = [cache[(context, sequence)] for sequence in frame["sequence"]]
        return frame

    def evaluate_variants(self, iteration_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates a series of sequences and returns a dataframe of the scores.
        """
        iteration_df = self._canonical_generator_scores(iteration_df)
        score_dfs = []
        for evaluator in self.evaluators:
            extra_kwargs = evaluator.get_extra_kwargs(self.base_variant)
            score_dfs.append(
                pd.DataFrame(
                    evaluator.evaluate(
                        iteration_df,
                        **extra_kwargs,
                    )
                )
            )

        # Concatenate the score dfs (all sequences are in the same order)
        full_df = pd.concat(score_dfs, axis=1) if score_dfs else pd.DataFrame(index=iteration_df.index)

        # Concatenate the generated population df to the score dfs
        full_df = pd.concat([iteration_df, full_df], axis=1)

        # Remove duplicate 'sequence' columns
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]

        # Return the full df
        return full_df

    def calculate_scores(self, iteration_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine finite oriented metrics. Bounds stay fixed for the entire run.

        With normalize=True, SCORE is a weighted mean in [0,1]. Without it,
        SCORE is a weighted mean of oriented raw metrics. SCORE_delta always
        records the signed improvement over the reference under that objective.
        """
        weights = self._fitness_weights()
        components, deltas = [], []
        for score_name, score_info in self.active_scores_dict.items():
            if score_name not in iteration_df:
                raise ValueError(f"Missing active score '{score_name}'")
            component = self._fitness_component(iteration_df[score_name], score_name, score_info)
            if getattr(self, "normalize", True):
                reference = self.base_variant.normalized_score_dict[score_name]
                iteration_df[score_name + "_normalized"] = component
            else:
                reference = self._fitness_component(
                    [self.base_variant.score_dict[score_name]], score_name, score_info
                )[0]
            self._finite_scores([reference], f"{score_name} reference")
            delta = component - reference
            iteration_df[score_name + "_delta"] = delta
            components.append(component)
            deltas.append(delta)
        iteration_df["SCORE"] = np.column_stack(components) @ weights
        iteration_df["SCORE_delta"] = np.column_stack(deltas) @ weights
        self._finite_scores(iteration_df["SCORE"], "SCORE")
        self._finite_scores(iteration_df["SCORE_delta"], "SCORE_delta")
        return iteration_df

    def _get_top_k(self, num_candidates: int) -> int:
        """Return the number of candidate rows retained before reconstruction."""
        return ceil(self.top_k_fraction * num_candidates)

    @abstractmethod
    def select_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        """
        Selects variants from the population from the evaluated population dataframe.

        Updates the selected count column in the dataframe to the number of times
        each variant was selected.

        Returns:
            pd.DataFrame containing the following columns (at least):
                - selected_count: The number of times each variant was selected
                - proposal_sequence: The sequence of the selected variant
                - proposal_SCORE: The score of the selected variant

        Implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def metropolis_hastings_filter(self, iteration_df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts or rejects variants based on the Metropolis-Hastings algorithm.

        Args:
            iteration_df : pd.DataFrame
                A dataframe with the following columns:
                    - sequence: The sequence of the variant
                    - starting_SCORE: The score of the variant before the Metropolis-Hastings filter
                    - SCORE: The score of the variant after the Metropolis-Hastings filter

        Returns:
            pd.DataFrame
                A dataframe with the following columns added:
                    - accepted: A boolean column indicating whether the variant was accepted
                    - proposal_sequence: The sequence of the accepted or rejected variant
                    - proposal_SCORE: The score of the accepted or rejected variant
        """

        # Get the temperature
        T = self.temperature_schedule(iteration=self.iteration)

        # Mark all variants as not accepted
        iteration_df["accepted"] = False

        # Calculate acceptance probabilities for selected variants
        scores = iteration_df["SCORE"]
        starting_scores = iteration_df["starting_SCORE"]
        if not np.isfinite(T) or T <= 0:
            raise ValueError("MH temperature must be finite and positive")
        if not np.isfinite(scores).all() or not np.isfinite(starting_scores).all():
            raise ValueError("MH scores must be finite")
        log_acceptance = np.minimum((scores - starting_scores) / T, 0.0)

        # Vectorized operation to determine acceptance
        random_values = np.random.rand(len(iteration_df))
        with np.errstate(divide="ignore"):
            iteration_df["accepted"] = np.log(random_values) < log_acceptance.values

        # Finally, add the proposal sequence and proposal score columns for selection
        iteration_df["proposal_sequence"] = np.where(
            iteration_df["accepted"],
            iteration_df["sequence"],
            iteration_df["starting_sequence"],
        )
        iteration_df["proposal_SCORE"] = np.where(
            iteration_df["accepted"],
            iteration_df["SCORE"],
            iteration_df["starting_SCORE"],
        )

        return iteration_df

    def update_population(self, iteration_df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Updates the population with the selected variants.
        """

        # Filter to only selected variants
        iteration_df = iteration_df[iteration_df["selected_count"] > 0].copy()

        # Create the new population
        population = []
        population_scores = []
        for index, row in iteration_df.iterrows():
            # Add the selected variants
            population.extend([row["proposal_sequence"]] * row["selected_count"])
            population_scores.extend([row["proposal_SCORE"]] * row["selected_count"])

        return (
            population,
            population_scores,
        )

    def _compute_context(self):
        """Return the compute context for this sampler's model calls.

        A ``ToolPool`` enters proto-tools' persistence context itself, so one
        block gives both warm workers and multi-GPU fan-out. Entering is a no-op
        when a pool is already active, which lets a caller own the pool and
        share it across several samplers — and is required, since proto-tools
        refuses nested pools.

        The pool takes every GPU it can see, so ``CUDA_VISIBLE_DEVICES``
        chooses where a run lands — including the empty string for a CPU run,
        which leaves the pool with no devices to reserve.

        Returns:
            A context manager: a new ``ToolPool`` or ``nullcontext``.
        """
        from proto_tools.utils import ToolPool
        from proto_tools.utils.tool_pool import _active_pool

        if _active_pool.get() is not None:
            return nullcontext()

        return ToolPool()

    def _release_resources(self, flush_cache=True):
        """Release writers and compute once, independently of export failures."""
        if getattr(self, "_resources_released", False):
            return []
        self._resources_released = True
        errors = []
        actions = []
        if flush_cache:
            actions.append(("CSV cache shutdown", shutdown_cache))
        stack = getattr(self, "_compute_stack", None)
        if stack is not None:
            actions.append(("compute context shutdown", stack.close))
        for label, action in actions:
            try:
                action()
            except BaseException as error:
                errors.append((label, error))
        return errors

    def _finish_tracking(self, status, errors):
        """Finish tracking after the caller has collected all other errors."""
        run = getattr(self, "wandb_run", None)
        if run is not None:
            try:
                run.finish(exit_code=0 if status == "success" and not errors else 1)
            except BaseException as error:
                errors.append(("W&B shutdown", error))

    def end_run(self, status: str = "success"):
        """Release workers and writers before independently saving diagnostics."""
        if getattr(self, "_ended", False):
            return
        self._ended = True
        errors = self._release_resources()
        output_dir = getattr(self, "output_dir", None)
        if output_dir is not None:
            actions = [
                ("log copy", lambda: logger.copy_log_file(dst_dir=output_dir)),
                ("reference score export", lambda: self.base_variant.save_yaml(output_dir / "base_variant_scores.yaml")),
            ]
            for label, action in actions:
                try:
                    action()
                except BaseException as error:
                    errors.append((label, error))
        self._finish_tracking(status=status, errors=errors)
        if errors:
            first_label, first_error = errors[0]
            first_error.add_note(f"Cleanup failed during {first_label}")
            for label, error in errors[1:]:
                first_error.add_note(f"Additional cleanup failure during {label}: {error}")
            raise first_error

    # ===== Main Loop =====
    def initialize_population(self):
        """Legacy optimization samplers start from copies of the reference."""
        reference = self.calculate_scores(pd.DataFrame([self.base_variant.score_dict]))
        initial = pd.concat([reference] * self.population_size, ignore_index=True)
        initial["sequence"] = self.base_variant.sequence
        initial["proposal_sequence"] = self.base_variant.sequence
        initial["proposal_SCORE"] = initial["SCORE"]
        initial["selected_count"] = 1
        initial["accepted"] = True
        self.iteration = -1
        try:
            self._cache_generations(initial, record_generated=False)
        finally:
            self.iteration = 0
        return (
            [self.base_variant.sequence] * self.population_size,
            [float(reference["SCORE"].iloc[0])] * self.population_size,
        )

    def run(self):
        """Run sampling and release resources without masking its original error."""
        if getattr(self, "_ended", False) or getattr(self, "_resources_released", False):
            raise RuntimeError("Sampler has already been closed")
        status, original_error = "success", None
        try:
            self._run()
        except BaseException as error:
            original_error = error
            status = "killed" if isinstance(error, KeyboardInterrupt) else "error"
            raise
        finally:
            try:
                self.end_run(status=status)
            except BaseException as cleanup_error:
                if original_error is None:
                    raise
                original_error.add_note(f"Cleanup also failed: {cleanup_error}")
                for note in getattr(cleanup_error, "__notes__", []):
                    original_error.add_note(note)

    def _run(self):
        """
        Contains core run logic loop for the sampler
        """

        logger.info(
            f"Running sampler with {self.population_size} population size and {self.num_iters} iterations"
        )

        # Determine min/max ranges for unspecified scores
        self._determine_min_max_ranges()

        # Initialize population dataframe
        population, population_scores = self.initialize_population()

        # Iterate
        for iteration_num in tqdm(
            range(self.num_iters),
            desc="Running sampler",
            total=self.num_iters,
            unit="Iteration",
        ):
            logger.info(f"Running iteration {iteration_num} of {self.num_iters}")

            # Explode the population by the number of variants per sequence
            iteration_df = pd.DataFrame(
                [
                    {"sequence": sequence, "starting_SCORE": score,
                     "parent_particle_id": parent_id}
                    for parent_id, (sequence, score) in enumerate(zip(population, population_scores))
                ]
            )

            # Mask sequences
            logger.info(f"Masking sequences for iteration {iteration_num}")
            iteration_df = self.mask_sequences(iteration_df=iteration_df)
            # Rename "sequence" to "starting_sequence"
            iteration_df.rename(columns={"sequence": "starting_sequence"}, inplace=True)

            # Generate variants
            logger.info(f"Generating variants for iteration {iteration_num}")
            iteration_df = self.generate_variants(iteration_df=iteration_df)

            # Evaluate variants
            self.run_stats["total_sequences_evaluated"] += len(iteration_df)
            logger.info(f"Evaluating variants for iteration {iteration_num}")
            iteration_df = self.evaluate_variants(iteration_df=iteration_df)

            # Calculate scores relative to the base variant
            logger.info(f"Calculating scores for iteration {iteration_num}")
            iteration_df = self.calculate_scores(iteration_df=iteration_df)

            # Initialize the selected count to 0
            iteration_df["selected_count"] = 0

            # Otherwise, select variants
            logger.info(f"Selecting variants for iteration {iteration_num}")
            iteration_df = self.select_variants(
                iteration_df=iteration_df,
                top_k=self._get_top_k(len(iteration_df)),
                num_to_select=self.population_size,
            )

            # Ensure that the selected count is equal to the population size
            if iteration_df["selected_count"].sum() != self.population_size:
                raise ValueError("The selected count is not equal to the population size")

            # Log trajectory
            logger.info(f"Caching trajectory for iteration {iteration_num}")
            self.cache_trajectory(iteration_df=iteration_df)

            # Update iteration
            self.iteration += 1

            # Update starting point with selected variants
            logger.info(f"Updating population for iteration {iteration_num}")
            population, population_scores = self.update_population(iteration_df=iteration_df)

        self._save_final_population(population, population_scores)


@SAMPLER_REGISTRY.register("rejection")
class Rejection(Sampler):

    def select_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        """
        Selects the top_k variants with the highest scores. If top_k is less than
        num_to_select, the remaining variants are selected in ranked order.
        """

        # No MH filter means all variants are accepted
        iteration_df["accepted"] = True
        iteration_df["proposal_sequence"] = iteration_df["sequence"]
        iteration_df["proposal_SCORE"] = iteration_df["SCORE"]

        # Sort the variants by score
        iteration_df = iteration_df.sort_values(
            by="proposal_SCORE", ascending=False
        ).reset_index(drop=True)

        # Distribute num_to_select across top_k variants
        base_count = num_to_select // top_k
        extra_count = num_to_select % top_k

        iteration_df.loc[: top_k - 1, "selected_count"] = base_count
        iteration_df.loc[: extra_count - 1, "selected_count"] += 1

        return iteration_df


def resample_systematic(idx, weights, K, rng):
    """Draw K indices directly from all non-negative particle weights."""
    idx = np.asarray(idx)
    w = np.asarray(weights, dtype=float)
    if K < 0 or int(K) != K:
        raise ValueError("Resample size must be a non-negative integer")
    K = int(K)
    if K == 0:
        return idx[:0]
    if w.ndim != 1 or len(w) != len(idx) or len(w) == 0:
        raise ValueError("Weights and indices must be nonempty and have equal length")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Resampling weights must be finite and non-negative")
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Resampling weights must have positive finite total mass")
    cdf = np.cumsum(w / w_sum)
    cdf[-1] = 1.0
    u0 = rng.random() / K
    u = u0 + (np.arange(K, dtype=float) / K)
    return idx[np.searchsorted(cdf, np.minimum(u, np.nextafter(1.0, 0.0)), side="right")]


@SAMPLER_REGISTRY.register("mada")
class MADA(Sampler):
    """
    MADA with stochastic and greedy top-K sampling policies.

    For a current-target invariant MH move, the reversed-kernel SMC weight is
    exp((beta_t - beta_{t-1}) * f(parent)), not f(the moved state).

    The first population is drawn from the generator with all designable sites
    masked (beta=0). Target-law validity assumes that partial-refill proposals
    are reversible with respect to that same generator prior; the SMC mechanics
    do not establish this property for a learned generator.

    greedy=True (the default) enables top-k selection and weighted population reconstruction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize RNG for reproducible resampling
        self.rng = np.random.default_rng(self.seed)

    def initialize_population(self):
        if self.greedy:
            return super().initialize_population()

        # A point mass of identical WT copies is not a sample from the prior.
        # Draw one independent completion per particle before annealing starts.
        fixed = set(self.base_variant.fixed_position_indices)
        prompt = "".join(
            residue if position in fixed else "_"
            for position, residue in enumerate(self.base_variant.sequence)
        )
        initial_df = pd.DataFrame({
            "prompt": [prompt] * self.population_size,
            "starting_sequence": [self.base_variant.sequence] * self.population_size,
            "starting_SCORE": [0.0] * self.population_size,
            "parent_particle_id": np.arange(self.population_size),
        })
        initial_df = self.generate_variants(initial_df, generations_per_sequence=1)
        if len(initial_df) != self.population_size:
            raise ValueError("SMC initialization requires one prior draw per particle")
        self.run_stats["total_sequences_evaluated"] += len(initial_df)
        initial_df = self.calculate_scores(self.evaluate_variants(initial_df))
        if not np.isfinite(initial_df["SCORE"]).all():
            raise ValueError("SMC initial scores must be finite")

        # Include prior draws in the lookup table and mark their trajectory as
        # initialization, before temperature step zero.
        initial_df["accepted"] = True
        initial_df["selected_count"] = 1
        initial_df["proposal_sequence"] = initial_df["sequence"]
        initial_df["proposal_SCORE"] = initial_df["SCORE"]
        self.iteration = -1
        try:
            self._cache_generations(initial_df)
        finally:
            self.iteration = 0
        return initial_df["sequence"].tolist(), initial_df["SCORE"].tolist()

    def _get_top_k(self, num_candidates: int) -> int:
        if self.greedy:
            return super()._get_top_k(num_candidates)

        requested_k = super()._get_top_k(num_candidates)
        if requested_k != self.population_size and not getattr(self, "_top_k_message_logged", False):
            logger.info(
                "top_k_fraction is ignored for stochastic sampling; "
                f"using K=N={self.population_size}."
            )
            self._top_k_message_logged = True
        return self.population_size

    def _select_greedy_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        """Select top-K retained states and reconstruct with annealed weights."""
        # Metropolis-Hastings
        iteration_df = self.metropolis_hastings_filter(iteration_df=iteration_df)

        # Get top_k variants and compute temperature values
        top_k_variants = iteration_df.nlargest(top_k, "proposal_SCORE", keep="first").copy()

        curr_T = self.temperature_schedule(iteration=self.iteration)
        prev_T = (
            self.temperature_schedule(iteration=self.iteration - 1)
            if self.iteration > 0
            else np.inf
        )

        # Compute annealed log-weights
        temp_factor = 1.0 / curr_T if prev_T == np.inf else (1.0 / curr_T - 1.0 / prev_T)
        logw = top_k_variants["proposal_SCORE"] * temp_factor

        # Convert to normalized probabilities
        weights = np.exp(logw - logw.max())  # Stabilized
        weights /= weights.sum()

        # Sample and count selections
        selected_indices = np.random.choice(
            top_k_variants.index, size=num_to_select, replace=True, p=weights
        )
        selection_counts = pd.Series(selected_indices).value_counts()

        # Update selected counts
        iteration_df.loc[selection_counts.index, "selected_count"] = selection_counts.values

        return iteration_df

    def select_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        if self.greedy:
            return self._select_greedy_variants(iteration_df, top_k, num_to_select)

        iteration_df = iteration_df.reset_index(drop=True).copy()

        if len(iteration_df) == 0:
            iteration_df["selected_count"] = 0
            return iteration_df

        if num_to_select < 1 or int(num_to_select) != num_to_select:
            raise ValueError("SMC population size must be a positive integer")
        curr_T = self.temperature_schedule(iteration=self.iteration)
        prev_T = (
            self.temperature_schedule(iteration=self.iteration - 1)
            if self.iteration > 0
            else np.inf
        )
        if not np.isfinite(curr_T) or curr_T <= 0 or prev_T <= 0 or np.isnan(prev_T):
            raise ValueError("SMC temperatures must be positive, with finite current temperature")
        temp_factor = 1.0 / curr_T - 1.0 / prev_T

        parent_scores = iteration_df["starting_SCORE"].to_numpy(dtype=float)
        if not np.isfinite(parent_scores).all():
            raise ValueError("SMC parent scores must be finite")
        logw = parent_scores * temp_factor
        if "parent_particle_id" in iteration_df:
            parents = iteration_df.groupby("parent_particle_id", sort=False)
            if (parents["starting_SCORE"].nunique() != 1).any():
                raise ValueError("All descendants of a particle must share its parent score")
            # Each ancestor's mass is shared by its descendants, so an ancestor
            # returning more designs does not acquire more importance weight.
            logw -= np.log(parents["starting_SCORE"].transform("size").to_numpy())
        weights = np.exp(logw - logw.max())

        # Batched moves are drawn before selection, but their outcomes must not
        # enter the reversed-kernel incremental weights.
        iteration_df = self.metropolis_hastings_filter(iteration_df=iteration_df)
        indices = iteration_df.index.to_numpy()
        if self.resampling_method == "multinomial":
            final_indices = self.rng.choice(
                indices, size=int(num_to_select), replace=True, p=weights / weights.sum()
            )
        else:
            final_indices = resample_systematic(
                indices, weights, int(num_to_select), self.rng
            )
        iteration_df["selected_count"] = 0
        counts = pd.Series(final_indices).value_counts()
        iteration_df.loc[counts.index, "selected_count"] = counts.to_numpy()
        return iteration_df
