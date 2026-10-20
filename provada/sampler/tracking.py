"""
tracking.py

Implements a mixin class for tracking the trajectory of the sampler.
"""

import pandas as pd
from provada.utils.cache import cache_to_csv
from provada.sequences.io import hash_sequence
from provada.sequences.mask import number_of_masked_positions


class TrackingMixin:
    """
    Mixin class for tracking the trajectory of the sampler.
    """

    def _sequence_columns(self):
        """Stable schema shared by sequence lookup and population exports."""
        columns = ["sequence", "SCORE", "SCORE_delta"]
        for name in self.active_scores_dict:
            columns.extend([name, name + "_delta"])
            if getattr(self, "normalize", True):
                columns.append(name + "_normalized")
        return columns

    def _cache_sequence_lookup(self, frame):
        """Register canonical per-sequence scores, including initial states."""
        if not hasattr(self, "_sequence_records"):
            self._sequence_records = {}
        new_records = []
        for record in frame.reindex(columns=self._sequence_columns()).to_dict("records"):
            key = hash_sequence(record["sequence"])
            previous = self._sequence_records.get(key)
            if previous is not None:
                for name in ["SCORE"] + list(self.active_scores_dict):
                    old, new = float(previous[name]), float(record[name])
                    if abs(old - new) > 1e-10 * max(1.0, abs(old), abs(new)):
                        raise ValueError(f"Fitness for sequence {key} changed within the run: {name}")
                continue
            record["sequence_hash"] = key
            self._sequence_records[key] = record
            new_records.append(record)
        if new_records:
            cache_to_csv(pd.DataFrame(new_records), self.output_dir / "sequence_lookup.csv")

    def _ensure_reference_lookup(self):
        if not getattr(self, "_reference_in_lookup", False):
            reference = self.calculate_scores(pd.DataFrame([self.base_variant.score_dict]))
            reference["sequence"] = self.base_variant.sequence
            self._cache_sequence_lookup(reference)
            self._reference_in_lookup = True

    def _cache_generations(self, iteration_df: pd.DataFrame, record_generated=True):
        """Save proposals separately from the actual post-acceptance states."""
        self._ensure_reference_lookup()
        self._cache_sequence_lookup(iteration_df)
        generated_hash = iteration_df["sequence"].map(hash_sequence)
        selected_hash = iteration_df["proposal_sequence"].map(hash_sequence)
        initial = self.iteration < 0
        parent_hash = (pd.Series(None, index=iteration_df.index, dtype=object) if initial
                       else iteration_df["starting_sequence"].map(hash_sequence))
        for key in list(selected_hash) + ([] if initial else list(parent_hash)):
            if key not in self._sequence_records:
                raise ValueError(f"Population state {key} is missing from the sequence lookup")

        if record_generated:
            new_hashes = set(generated_hash) - self.sequence_hashes
            rows = iteration_df.loc[generated_hash.isin(new_hashes)].drop_duplicates("sequence")
            if len(rows):
                cache_to_csv(rows.reindex(columns=self._sequence_columns()),
                             self.output_dir / "generated_sequences.csv")
            self.sequence_hashes.update(new_hashes)
            self.run_stats["num_unique_sequences"] += len(new_hashes)

        trajectory = pd.DataFrame({
            "trajectory_schema_version": 2,
            "iteration": self.iteration,
            "parent_particle_id": None if initial else iteration_df.get("parent_particle_id"),
            "parent_sequence_hash": parent_hash,
            "generated_sequence_hash": generated_hash,
            "selected_sequence_hash": selected_hash,
            # Compatibility key now consistently identifies the retained state.
            "sequence_hash": selected_hash,
            "parent_SCORE": None if initial else iteration_df["starting_SCORE"],
            "generated_SCORE": iteration_df["SCORE"],
            "selected_SCORE": iteration_df["proposal_SCORE"],
            "selected_count": iteration_df["selected_count"],
            "accepted": iteration_df["accepted"],
            "generation_mpnn_score": iteration_df.get("generation_mpnn_score"),
        })
        cache_to_csv(trajectory, self.output_dir / "trajectory.csv")

    def _selected_population_frame(self, iteration_df):
        """Expand retained states in exactly the same order as update_population."""
        rows = []
        for row in iteration_df.itertuples(index=False):
            count = row.selected_count
            if int(count) != count or count < 0:
                raise ValueError("Population multiplicities must be nonnegative integers")
            if count:
                record = self._sequence_records[hash_sequence(row.proposal_sequence)]
                if abs(float(record["SCORE"]) - row.proposal_SCORE) > 1e-10 * max(1.0, abs(row.proposal_SCORE)):
                    raise ValueError("Retained sequence and score do not match the sequence lookup")
                rows.extend([record] * int(count))
        return pd.DataFrame(rows)

    def _save_final_population(self, population, population_scores):
        """Write all N particles, preserving their order and duplicate sequences."""
        if len(population) != self.population_size or len(population_scores) != len(population):
            raise ValueError("Final population must contain exactly population_size particles")
        rows = []
        for particle_id, (sequence, score) in enumerate(zip(population, population_scores)):
            record = self._sequence_records[hash_sequence(sequence)].copy()
            if abs(float(record["SCORE"]) - score) > 1e-10 * max(1.0, abs(score)):
                raise ValueError("Final particle score does not match its sequence")
            record.update(particle_id=particle_id, selected_count=1)
            rows.append(record)
        self.final_population = pd.DataFrame(rows)
        self.final_population.to_csv(self.output_dir / "final_population.csv", index=False)

    def _wandb_log(self, iteration_df: pd.DataFrame):
        """
        Logs the trajectory of the sampler to wandb.
        """

        # Create a log dict to store values we want to log to wandb
        log_dict = {}

        # VALUE STATISTICS ======================================================
        # Calculate value statistics for the current population
        value_statistics = self._selected_population_frame(iteration_df).describe()

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
        log_dict["num_masked_sites"] = number_of_masked_positions(
            self.masking_schedule(iteration=self.iteration),
            self.base_variant.num_designable_positions,
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
