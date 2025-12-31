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

        # Mask sequences
        reference_dist = []
        for _ in range(number_of_sequences):
            # Pick a random number of masked sites
            num_masked_sites = np.random.randint(
                1,
                min(
                    int(round(self.base_variant.sequence_length * 0.7)),
                    self.base_variant.num_designable_positions,
                ),
            )
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
            columns_to_keep = ["sequence"] + [
                col for col in score_df.columns if col in score_subset
            ]
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
        scores_with_unspecified_ranges = []
        for score_name, score_info in self.active_scores_dict.items():
            # Save the specified min/max values for tracking
            specified_min_value = score_info["min_value"]
            if specified_min_value is None:
                specified_min_value = "None"

            self.active_scores_dict[score_name]["specified_min_value"] = specified_min_value

            specified_max_value = score_info["max_value"]
            if specified_max_value is None:
                specified_max_value = "None"

            self.active_scores_dict[score_name]["specified_max_value"] = specified_max_value

            # If the min/max values are not specified, add the score to the list of scores with unspecified ranges
            if not isinstance(score_info["min_value"], (int, float)):
                scores_with_unspecified_ranges.append(
                    (score_name, "min_value", score_info["min_value"])
                )

            if not isinstance(score_info["max_value"], (int, float)):
                scores_with_unspecified_ranges.append(
                    (score_name, "max_value", score_info["max_value"])
                )

        # Get the reference distribution
        if len(scores_with_unspecified_ranges) > 0:
            reference_dist = self._get_reference_distribution(scores_with_unspecified_ranges)

        # Update the min/max ranges for the scores with unspecified ranges
        for score_name, value_type, specified in scores_with_unspecified_ranges:
            if isinstance(specified, str):
                self.active_scores_dict[score_name][value_type] = getattr(
                    self.base_variant, specified
                )
            elif specified is None:
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
            else:
                raise ValueError(f"Invalid type for specified value: {specified}")

        # Determine base variant normalized scores
        self.base_variant.normalized_score_dict = {}
        for score_name, score_info in self.active_scores_dict.items():
            self.base_variant.normalized_score_dict[score_name] = (
                self.base_variant.score_dict[score_name] - score_info["min_value"]
            ) / (score_info["max_value"] - score_info["min_value"])

            # Ensure it is in the range
            if abs(self.base_variant.normalized_score_dict[score_name]) > 1:
                raise ValueError(
                    f"BaseVariant should not be out of range after range determination"
                )

            # Flip score if larger is not better
            if not score_info["larger_is_better"]:
                self.base_variant.normalized_score_dict[score_name] = (
                    1 - self.base_variant.normalized_score_dict[score_name]
                )

        # Log active scores to wandb
        if self.wandb_run is not None:
            self.wandb_run.summary.update({"active_scores_dict": self.active_scores_dict})
        logger.info(f"Final active scores dict with min/max ranges: {self.active_scores_dict}")
