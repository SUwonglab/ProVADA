"""
sampler.py

Implements a class for Samplers, the primary construct for the ProVADA codebase.
Samplers are responsible for calling other components and directing the flow of
the search space.
"""

import os
import numpy as np
from pathlib import Path
from math import floor, ceil
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from omegaconf import OmegaConf

from typing import Dict, List, Any, Optional, Tuple
from provada.base_variant import BaseVariant
from provada.utils.log import get_logger
from provada.utils.env import device_manager
from provada.utils.setup import seed_everything
from provada.utils.registry import SAMPLER_REGISTRY
from provada.utils.cache import shutdown_cache
from provada.sampler.tracking import TrackingMixin
from provada.sampler.startup import StartupMixin
from provada.sampler.reference_ranges import ReferenceRangesMixin
import wandb
import time
import yaml


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
        masking_strategy_type: str = "random",
        masking_strategy_kwargs: Optional[Dict[str, Any]] = None,
        masking_schedule_config: Optional[Dict[str, Any]] = None,
        temperature_schedule_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        device: Optional[str] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        reference_distribution_file: Optional[str] = None,
        wandb_run: Optional[wandb.Run] = None,
        config: Optional[Dict[str, Any]] = None,
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
            masking_strategy_type: The type of masking strategy to use.
            masking_strategy_kwargs: The kwargs to pass to the masking strategy.
            masking_schedule_config: The configuration for the masking schedule.
            temperature_schedule_config: The configuration for the temperature schedule.
            seed: Seed to use for various components of the sampler.
            device: The device to use for the sampler.
            run_name: The name of the run.
            output_dir: The directory to save the results to. If not provided, defaults to "results".
            reference_distribution_file: The file to load the reference distribution from. If not provided, defaults to None.
            wandb_run: The wandb run to use. If not provided, defaults to None.
            config: Optional config to save to the output directory. NOTE: The
                config will not be used for anything other than saving to the output directory.

        """
        # Set input attributes
        self.base_variant = base_variant
        self.num_iters = num_iters
        self.population_size = population_size
        self.top_k_fraction = top_k_fraction
        self.seed = seed
        self.active_scores_dict = {
            score: {"weight": weight} for score, weight in score_weights.items()
        }
        self.wandb_run = wandb_run
        self.reference_distribution_file = reference_distribution_file

        # Set device manager attributes
        device_manager.set_device(device)

        # Seed everything
        seed_everything(self.seed)

        # Evaluator set up
        self.evaluators = []
        self.initialize_evaluators()

        # Generator set up
        self.generation_kwargs = generation_kwargs
        self.generator = None
        self.initialize_generator(generator_type, generation_kwargs)

        # Masking strategy set up
        self.masking_strategy = None
        self.initialize_masking_strategy(masking_strategy_type, masking_strategy_kwargs)

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
            with open(
                self.output_dir / f"{run_name}_config.yaml",
                "w",
            ) as f:
                yaml.dump(OmegaConf.to_container(config, resolve=True), f)

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

    # ===== Helper Functions =====
    def mask_sequences(self, iteration_df: pd.DataFrame):
        """
        Masks the sequences using the masking strategy.
        """
        return self.masking_strategy.create_masked_sequences(
            sequences=iteration_df,
            num_masked_sites=max(
                floor(
                    self.masking_schedule(iteration=self.iteration)
                    * self.base_variant.num_designable_positions
                ),
                1,  # Always make sure there is at least one masked site
            ),
        )

    def generate_variants(self, iteration_df: pd.DataFrame):
        """
        Generates a series of sequences using the generator.
        """
        if isinstance(self.generation_kwargs, dict):
            generate_kwargs = self.generation_kwargs
        else:
            generate_kwargs = OmegaConf.to_container(self.generation_kwargs, resolve=True)

        # Get any extra kwargs needed by this generator
        extra_kwargs = self.generator.get_extra_kwargs(self.base_variant)
        generate_kwargs.update(extra_kwargs)

        return self.generator.generate(
            iteration_df,
            **generate_kwargs,
        )

    def evaluate_variants(self, iteration_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates a series of sequences and returns a dataframe of the scores.
        """
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
        full_df = pd.concat(score_dfs, axis=1)

        # Concatenate the generated population df to the score dfs
        full_df = pd.concat([iteration_df, full_df], axis=1)

        # Remove duplicate 'sequence' columns
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]

        # Return the full df
        return full_df

    def calculate_scores(self, iteration_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the scores relative to the base variant. If the score is lower
        than the base variant, it is set to 0.
        """
        for score_name, score_info in self.active_scores_dict.items():

            # Get the base variant normalized score
            base_variant_normalized_score = self.base_variant.normalized_score_dict[score_name]

            # Scale the score. Clip to be between 0 and 1.
            iteration_df[score_name + "_normalized"] = (
                iteration_df[score_name]
                .sub(score_info["min_value"])
                .div(score_info["max_value"] - score_info["min_value"])
            ).clip(lower=0, upper=1)

            # Flip score if larger is not better
            if not score_info["larger_is_better"]:
                iteration_df[score_name + "_normalized"] = (
                    1 - iteration_df[score_name + "_normalized"]
                )

            # Calculate the delta of the scores by subtracting the base variant normalized score
            iteration_df[score_name + "_delta"] = iteration_df[score_name + "_normalized"].sub(
                base_variant_normalized_score
            )

        # Calculate the total score as a weighted sum of the deltas
        delta_cols = [score_name + "_delta" for score_name in self.active_scores_dict.keys()]
        weights_dict = {
            score_delta: self.active_scores_dict[score_delta.replace("_delta", "")]["weight"]
            for score_delta in delta_cols
        }
        # Calculate sum of weights
        sum_of_weights = sum(weights_dict.values())

        iteration_df["SCORE"] = (
            iteration_df.loc[:, delta_cols].mul(weights_dict, axis=1).sum(axis=1)
        ).div(sum_of_weights)

        return iteration_df

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
        acceptance_probabilities = np.exp((scores - starting_scores) / T)

        # Vectorized operation to determine acceptance
        random_values = np.random.rand(len(iteration_df))
        iteration_df["accepted"] = random_values < acceptance_probabilities.values

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

    def end_run(self, status: str = "success"):
        """
        Executes at the end of a run of the sampler
        """

        # Shutdown the cache
        shutdown_cache()

        # End the wandb run
        if self.wandb_run is not None:
            if status == "success":
                self.wandb_run.finish(exit_code=0)
            elif status == "killed":
                pass  # We don't need to do anything here, wandb automatically detects for killed runs
            elif status == "error":
                self.wandb_run.finish(exit_code=1)

        # Copy the log file to the output directory
        logger.copy_log_file(dst_dir=self.output_dir)

        # Save the base variant scores to the output directory
        self.base_variant.save_yaml(self.output_dir / "base_variant_scores.yaml")

    # ===== Main Loop =====
    def run(self):
        """
        Runs the sampler
        """
        status = "success"
        try:
            self._run()
        except KeyboardInterrupt as e:
            logger.info(f"Sampler interrupted by user during iteration {self.iteration}")
            status = "killed"
            raise e
        except Exception as e:
            logger.error(f"Error running sampler during iteration {self.iteration}: {e}")
            status = "error"
            raise e
        finally:
            try:
                self.end_run(status=status)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

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
        population = [self.base_variant.sequence] * self.population_size
        population_scores = [0] * self.population_size

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
                    {"sequence": sequence, "starting_SCORE": score}
                    for sequence, score in zip(population, population_scores)
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

            # Update masking strategy
            self.masking_strategy.update(
                masked_strings=iteration_df["prompt"].tolist(),
                rewards=iteration_df["SCORE"].tolist(),
            )

            # Initialize the selected count to 0
            iteration_df["selected_count"] = 0

            # Otherwise, select variants
            logger.info(f"Selecting variants for iteration {iteration_num}")
            iteration_df = self.select_variants(
                iteration_df=iteration_df,
                top_k=ceil(self.top_k_fraction * len(iteration_df)),
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


@SAMPLER_REGISTRY.register("mada_greedy")
class MADAGreedy(Sampler):

    def select_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        """
        Selects the top_k variants with the highest scores. If top_k is less than
        num_to_select, the remaining variants are selected in ranked order.
        """
        # Metropolis-Hastings
        iteration_df = self.metropolis_hastings_filter(iteration_df=iteration_df)

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


@SAMPLER_REGISTRY.register("mada_topk_is")
class MADATopKIS(Sampler):

    def select_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        """
        Greedy selection: deterministically selects top_k variants by score,
        then uses temperature-weighted resampling among those elites to reach num_to_select.

        This is a hybrid approach that combines deterministic elite selection
        with probabilistic resampling based on annealed weights.
        """
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


def resample_systematic(idx, weights, K, rng):
    w = np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        w = np.ones_like(w) / w.size
    else:
        w = w / w_sum
    cdf = np.cumsum(w)
    u0 = rng.random() / K
    u = u0 + (np.arange(K, dtype=float) / K)
    return idx[np.searchsorted(cdf, u, side="left")]


@SAMPLER_REGISTRY.register("mada_stochastic")
class MADAStochastic(Sampler):
    """
    Two stage resampling scheme for MADA with systematic resampling for the first stage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize RNG for reproducible resampling
        self.rng = np.random.default_rng(self.seed)

    def select_variants(
        self, iteration_df: pd.DataFrame, top_k: int, num_to_select: int
    ) -> pd.DataFrame:
        iteration_df = self.metropolis_hastings_filter(iteration_df=iteration_df)

        if len(iteration_df) == 0:
            iteration_df["selected_count"] = 0
            return iteration_df

        top_k = max(1, min(int(top_k), len(iteration_df)))
        num_to_select = max(1, int(num_to_select))
        curr_T = self.temperature_schedule(iteration=self.iteration)
        prev_T = (
            self.temperature_schedule(iteration=self.iteration - 1)
            if self.iteration > 0
            else np.inf
        )
        temp_factor = (1.0 / curr_T) if prev_T == np.inf else (1.0 / curr_T - 1.0 / prev_T)

        s = iteration_df["proposal_SCORE"].to_numpy()
        logw = s * temp_factor
        logw = np.where(np.isfinite(logw), logw, -np.inf)
        w = np.exp(logw - np.nanmax(logw))

        if not np.isfinite(w).any() or w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()

        idx = iteration_df.index.to_numpy()
        elite_indices = resample_systematic(idx, w, top_k, self.rng)
        final_indices = self.rng.choice(elite_indices, size=num_to_select, replace=True)
        iteration_df["selected_count"] = 0
        counts = pd.Series(final_indices).value_counts()
        iteration_df.loc[counts.index, "selected_count"] = counts.to_numpy()
        return iteration_df
