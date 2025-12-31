"""
tracking.py

Implements a mixin class for tracking the trajectory of the sampler.
"""

import pandas as pd
from provada.utils.cache import cache_to_csv
from provada.sequences.io import hash_sequence
from math import floor


class TrackingMixin:
    """
    Mixin class for tracking the trajectory of the sampler.
    """

    def _cache_generations(self, iteration_df: pd.DataFrame):
        """
        Caches the generations to a csv file and updates the number of unique sequences.
        """
        # Get sequence hashes of the evaluated population
        iteration_df["sequence_hash"] = iteration_df["sequence"].map(hash_sequence)

        # Generation Cache ---
        # All generated sequences are added once to the generated_sequences cache

        # Get the sequences hashes that are new
        new_sequence_hashes = set(iteration_df["sequence_hash"]) - set(
            self.sequence_hashes
        )
        # Update the number of unique sequences
        self.run_stats["num_unique_sequences"] += len(new_sequence_hashes)

        # Pull only the rows that are new
        new_iteration_df = iteration_df[
            iteration_df["sequence_hash"].isin(new_sequence_hashes)
        ].drop_duplicates(subset="sequence_hash")

        # Drop the columns that are not needed to save space
        new_iteration_df = new_iteration_df.drop(
            columns=[
                "prompt",
                "starting_sequence",
                "selected_count",
                "accepted",
                "starting_SCORE",
                "sequence_hash",
                "proposal_sequence",
                "proposal_SCORE",
            ]
        )

        # Cache the new rows
        if len(new_iteration_df) > 0:
            cache_to_csv(new_iteration_df, self.output_dir / "generated_sequences.csv")

        # Add the new sequence hashes to the sequence hashes set
        self.sequence_hashes.update(new_sequence_hashes)

        # Trajectory Cache ---
        # All sequence hashes are added each iteration to the trajectory cache
        # to track the history of the sampler
        iteration_df["iteration"] = self.iteration
        trajectory_df = iteration_df[
            ["sequence_hash", "selected_count", "accepted", "iteration"]
        ]
        # Cast accepted to int to save space
        cache_to_csv(trajectory_df, self.output_dir / "trajectory.csv")

        # Log position stats to csv
        position_stats_row = self.masking_strategy.get_position_stats_row_form()
        position_stats_row["iteration"] = self.iteration
        cache_to_csv(
            position_stats_row,
            self.output_dir / "position_stats.csv",
        )

    def _wandb_log(self, iteration_df: pd.DataFrame):
        """
        Logs the trajectory of the sampler to wandb.
        """

        # Create a log dict to store values we want to log to wandb
        log_dict = {}

        # VALUE STATISTICS ======================================================
        # Calculate value statistics for the current population
        value_statistics = iteration_df.describe()

        # Drop the 'count' row
        value_statistics = value_statistics.drop(index="count")

        # Add value statistics
        for score in value_statistics:
            if (
                "selected_count" not in score
                and "iteration" not in score
                and "accepted" not in score
                and "delta" not in score
                and "starting_SCORE" not in score
                and "proposal_SCORE" not in score
            ):
                log_dict[score + ".iteration_mean"] = value_statistics.loc[
                    "mean", score
                ]
                log_dict[score + ".iteration_min"] = value_statistics.loc["min", score]
                log_dict[score + ".iteration_max"] = value_statistics.loc["max", score]

        # Also add max values to log dict
        for score in self.active_scores_dict.keys():
            log_dict[score + ".all_time_max"] = self.observed_value_stats[score]["max"]
            log_dict[score + ".all_time_min"] = self.observed_value_stats[score]["min"]

        # Add the number of unique sequences and total sequences evaluated
        log_dict.update(self.run_stats)

        # Add 'iteration' as 'step'
        log_dict["step"] = self.iteration

        # Include scheduled values
        log_dict["mh_temperature"] = self.temperature_schedule(iteration=self.iteration)
        log_dict["percent_masked"] = self.masking_schedule(iteration=self.iteration)
        log_dict["num_masked_sites"] = floor(
            self.masking_schedule(iteration=self.iteration)
            * self.base_variant.num_designable_positions
        )

        # Add the fraction of MH acceptances
        log_dict["fraction_mh_acceptances"] = iteration_df["accepted"].sum() / len(
            iteration_df
        )

        # Log the table to wandb
        self.wandb_run.log(log_dict)

    def cache_trajectory(self, iteration_df: pd.DataFrame):
        """
        Logs the trajectory of the sampler.
        """

        # Update the observed value stats
        self._update_observed_value_stats(iteration_df=iteration_df)

        # Cache the generations
        self._cache_generations(iteration_df=iteration_df)

        # Log values to wandb
        if self.wandb_run is not None:
            self._wandb_log(iteration_df=iteration_df)

    def _update_observed_value_stats(self, iteration_df: pd.DataFrame):
        """
        Updates the max values for the scores.
        """
        # Update the max and min value
        for score_name in list(self.active_scores_dict.keys()) + ["SCORE"]:
            self.observed_value_stats[score_name]["max"] = max(
                self.observed_value_stats[score_name]["max"],
                iteration_df[score_name].max(),
            )
            self.observed_value_stats[score_name]["min"] = min(
                self.observed_value_stats[score_name]["min"],
                iteration_df[score_name].min(),
            )
