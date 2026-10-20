"""
reference_ranges.py

Contains the ReferenceRanges mixin class that helps determine the min/max ranges for the scores.
"""

import pandas as pd
from provada.utils.log import get_logger
import numpy as np
from provada.sequences.mask import mask_k
from typing import List, Tuple, Any

logger = get_logger(__name__)


class ReferenceRangesMixin:
    """
    Contains the reference ranges for the scores.
    """

    def _get_reference_distribution(
        self,
        scores_with_unspecified_ranges: List[Tuple[str, str, Any]],
        number_of_sequences: int = 2000,
    ) -> pd.DataFrame:
        """
        Creates a reference distribution of sequences to get a sense for the general
        range of scores for sequences.

        Only evaluates the scores that have unspecified ranges to avoid unnecessary
        computation.
        """

        # If reference distribution file was specified, load it in
        if self.reference_distribution_file is not None:
            logger.info(
                f"Loading reference distribution from {self.reference_distribution_file}"
            )
            reference_dist_file = self.reference_distribution_file
            reference_dist = pd.read_csv(str(reference_dist_file))
            return reference_dist

        # Determine which scores need to be calculated
        scores_needing_ranges = set(
            score_name for score_name, _, _ in scores_with_unspecified_ranges
        )
        logger.info(
            f"Creating reference distribution for scores: {scores_needing_ranges} "
            f"with {number_of_sequences} sequences"
        )

        designable = self.base_variant.num_designable_positions
        if designable < 1:
            raise ValueError("Reference calibration requires at least one designable residue")
        max_masked_sites = min(
            max(1, int(round(self.base_variant.sequence_length * 0.7))), designable
        )

        # Mask between one and max_masked_sites positions, inclusively.
        reference_dist = []
        for _ in range(number_of_sequences):
            num_masked_sites = np.random.randint(1, max_masked_sites + 1)
            # Mask the sequence
            masked_sequence = mask_k(
                sequence=self.base_variant.sequence,
                k=num_masked_sites,
                mask_str="_",
                fixed_indices=self.base_variant.fixed_position_indices,
            )
            reference_dist.append(masked_sequence)

        # Create a dataframe of the reference distribution
        reference_dist = pd.DataFrame({"prompt": reference_dist})

        # Generate the reference distribution
        reference_dist = self.generate_variants(iteration_df=reference_dist)

        # Calculate scores - but only for scores that need reference ranges
        reference_dist = self._evaluate_variants_subset(
            iteration_df=reference_dist,
            score_subset=scores_needing_ranges,
        )

        # Save the reference distribution to a CSV file
        reference_dist.drop(columns=["prompt"], inplace=True)
        reference_dist.to_csv(self.output_dir / "reference_distribution.csv", index=False)

        return reference_dist

    def _evaluate_variants_subset(
        self, iteration_df: pd.DataFrame, score_subset: set
    ) -> pd.DataFrame:
        """
        Evaluates a series of sequences but only for a subset of scores.

        This is useful for optimizing performance when only certain scores need
        to be calculated (e.g., when determining reference ranges).

        Args:
            iteration_df: The dataframe containing sequences to evaluate
            score_subset: A set of score names to evaluate

        Returns:
            The dataframe with sequences and the requested scores
        """
        logger.debug(
            f"Evaluating variants for score subset: {score_subset} "
            f"(skipping {len(self.active_scores_dict) - len(score_subset)} scores)"
        )

        score_dfs = []
        evaluators_used = 0
        iteration_df = self._canonical_generator_scores(iteration_df, score_subset)

        for evaluator in self.evaluators:
            # Check if this evaluator produces any scores in the subset
            evaluator_scores = set(evaluator.active_scores)
            needed_scores = evaluator_scores.intersection(score_subset)

            if not needed_scores:
                # Skip this evaluator - it doesn't produce any needed scores
                logger.debug(
                    f"Skipping evaluator {evaluator.__class__.__name__} "
                    f"(produces {evaluator_scores}, none in {score_subset})"
                )
                continue

            evaluators_used += 1
            logger.debug(
                f"Running evaluator {evaluator.__class__.__name__} "
                f"for scores {needed_scores}"
            )

            extra_kwargs = evaluator.get_extra_kwargs(self.base_variant)

            score_df = pd.DataFrame(
                evaluator.evaluate(
                    iteration_df,
                    **extra_kwargs,
                )
            )

            # Only keep the columns we actually need
            columns_to_keep = [col for col in score_df.columns if col in score_subset]
            score_df = score_df[columns_to_keep]

            score_dfs.append(score_df)

        logger.info(
            f"Evaluated {len(score_subset)} scores using {evaluators_used}/{len(self.evaluators)} evaluators "
            f"(saved {len(self.evaluators) - evaluators_used} evaluator calls)"
        )

        # Concatenate the score dfs (all sequences are in the same order)
        if score_dfs:
            full_df = pd.concat(score_dfs, axis=1)
        else:
            # No evaluators ran, just return the input with sequence column
            full_df = pd.DataFrame()

        # Concatenate the generated population df to the score dfs
        full_df = pd.concat([iteration_df, full_df], axis=1)

        # Remove duplicate 'sequence' columns
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]

        # Return the full df
        return full_df

    def _determine_min_max_ranges(self):
        """
        Determines the min/max ranges for scores where min/max values are not
        specified.
        """
        for name in self.active_scores_dict:
            if name not in self.base_variant.score_dict:
                raise ValueError(f"Missing active reference score '{name}'")
            self._finite_scores([self.base_variant.score_dict[name]], f"{name} reference")
        if not getattr(self, "normalize", True):
            self.base_variant.normalized_score_dict = {}
            return

        scores_with_unspecified_ranges = []
        for score_name, score_info in self.active_scores_dict.items():
            for value_type in ("min_value", "max_value"):
                specified = score_info[value_type]
                score_info["specified_" + value_type] = "None" if specified is None else specified
                if isinstance(specified, str):
                    # Attribute bounds are already known; they need no model calls.
                    score_info[value_type] = getattr(self.base_variant, specified)
                elif specified is None:
                    scores_with_unspecified_ranges.append((score_name, value_type, None))
                elif not isinstance(specified, (int, float)):
                    raise ValueError(f"Invalid type for specified value: {specified}")

        # Get the reference distribution
        if len(scores_with_unspecified_ranges) > 0:
            reference_dist = self._get_reference_distribution(scores_with_unspecified_ranges)
            for name in {name for name, _, _ in scores_with_unspecified_ranges}:
                if name not in reference_dist or reference_dist.empty:
                    raise ValueError(f"No reference distribution for active score '{name}'")
                self._finite_scores(reference_dist[name], f"{name} reference distribution")

        # Update the min/max ranges for the scores with unspecified ranges
        for score_name, value_type, _ in scores_with_unspecified_ranges:
            if value_type == "min_value":
                self.active_scores_dict[score_name][value_type] = min(
                    reference_dist[score_name].quantile(0.25),
                    self.base_variant.score_dict[score_name],
                )
            elif value_type == "max_value":
                self.active_scores_dict[score_name][value_type] = max(
                    reference_dist[score_name].quantile(0.75),
                    self.base_variant.score_dict[score_name],
                )
            else:
                raise ValueError(f"Invalid value type: {value_type}")

        # Sparse or discrete metrics can have tied quartiles despite varying
        # across calibration sequences. Widen only estimated bounds to the
        # observed range; explicitly supplied bounds remain unchanged.
        for score_name in {name for name, _, _ in scores_with_unspecified_ranges}:
            score_info = self.active_scores_dict[score_name]
            if score_info["min_value"] != score_info["max_value"]:
                continue
            estimated_bounds = {
                value_type for name, value_type, _ in scores_with_unspecified_ranges
                if name == score_name
            }
            if "min_value" in estimated_bounds:
                score_info["min_value"] = float(min(
                    reference_dist[score_name].min(), score_info["min_value"]
                ))
            if "max_value" in estimated_bounds:
                score_info["max_value"] = float(max(
                    reference_dist[score_name].max(), score_info["max_value"]
                ))
            if score_info["max_value"] <= score_info["min_value"]:
                raise ValueError(
                    f"Active score '{score_name}' has no calibration spread; provide "
                    "explicit finite bounds with max_value > min_value or set normalize=False"
                )
            logger.warning(
                f"Tied calibration quartiles for '{score_name}'; using observed bounds "
                f"[{score_info['min_value']}, {score_info['max_value']}]"
            )

        # Determine base variant normalized scores
        self.base_variant.normalized_score_dict = {}
        for score_name, score_info in self.active_scores_dict.items():
            self.base_variant.normalized_score_dict[score_name] = float(
                self._fitness_component([self.base_variant.score_dict[score_name]], score_name, score_info)[0]
            )

        # Log active scores to wandb
        if self.wandb_run is not None:
            self.wandb_run.summary.update({"active_scores_dict": self.active_scores_dict})
        logger.info(f"Final active scores dict with min/max ranges: {self.active_scores_dict}")
