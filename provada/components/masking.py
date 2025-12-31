"""
masking.py

Contains the MaskingStrategy class, which is used to select positions in a
sequence to mask for the next generation.
"""

import pandas as pd
from typing import List, Optional, Union
import numpy as np
from abc import ABC, abstractmethod
from provada.sequences.mask import mask_assigned_positions, mask_k
from provada.utils.registry import MASK_STRATEGY_REGISTRY


def get_masking_strategy(masking_strategy_type: str, **kwargs) -> "MaskingStrategy":
    """
    Returns a masking strategy of the given type.
    """
    return MASK_STRATEGY_REGISTRY.get_instance(masking_strategy_type, **kwargs)


class MaskingStrategy(ABC):
    """
    Selects positions in a sequence to make designable in the next generation.

    Saves statistics on which positions have been masked and designed.
    """

    def __init__(
        self,
        sequence_length: int,
        fixed_position_indices: List[int] = None,
        mask_char: str = "_",
    ):
        # Set attributes
        self.sequence_length = sequence_length
        self.mask_char = mask_char

        # Set fixed position indices
        if fixed_position_indices is None:
            self.fixed_position_indices = []
        else:
            # Sort fixed position indices and ensure they are within the sequence length
            self.fixed_position_indices = sorted(fixed_position_indices)
            if self.fixed_position_indices[0] < 0:
                raise ValueError(
                    f"Fixed position indices must be greater than 0. "
                    f"Got {self.fixed_position_indices[0]} < 0"
                )
            if self.fixed_position_indices[-1] >= self.sequence_length:
                raise ValueError(
                    f"Fixed position indices must be less than the sequence length. "
                    f"Got {self.fixed_position_indices[-1]} >= {self.sequence_length}"
                )

        # Set designable positions
        self.designable_positions = [
            i for i in range(self.sequence_length) if i not in self.fixed_position_indices
        ]

        # Set up statistics for designable positions
        self.position_stats = {}
        self.number_of_updates = 0
        self.number_of_samples = 0
        self.initialize(position_stat_defaults={"selection_count": 0, "reward": 0})

    def initialize(self, position_stat_defaults: dict = None):
        """
        Initializes the position statistics for the designable positions.
        """
        # Reset number of updates
        self.number_of_updates = 0
        self.number_of_samples = 0

        # Set defaults
        position_stat_defaults = position_stat_defaults or {}
        # Always include count in the defaults
        if "selection_count" not in position_stat_defaults:
            position_stat_defaults["selection_count"] = 0

        # Initialize the position statistics
        self.position_stats = {}

        for pos in self.designable_positions:
            self.position_stats[pos] = position_stat_defaults.copy()

    def update(self, masked_strings: List[str], rewards: Optional[List[float]] = None):
        """
        Updates the statistics of the masker based on the masked strings and rewards.

        Args:
            masked_strings (List[str]): A list of strings containing the masked positions.
            rewards (Optional[List[float]]): A list of rewards for the masked strings.
                Should be the same length as masked_strings.
        """
        # Increment number of updates
        self.number_of_updates += 1
        self.number_of_samples += len(masked_strings)

        # For each masked string
        for string_ind, masked_string in enumerate(masked_strings):

            # Determine masked positions
            masked_positions = [
                pos for pos, char in enumerate(masked_string) if char == self.mask_char
            ]

            # Increment selection count for each masked position
            for pos in masked_positions:
                self.position_stats[pos]["selection_count"] += 1

            # Update position stats
            self._update_position_stats(masked_positions, rewards[string_ind])

    def _update_position_stats(self, masked_positions: int, reward_for_string: float):
        """
        Updates the reward values at each designable position based on the masked
        positions and the reward for the string.
        """
        per_position_reward = (
            reward_for_string / len(masked_positions) if len(masked_positions) > 0 else 0
        )

        # In default implementation, we just add the reward
        for pos in masked_positions:
            self.position_stats[pos]["reward"] += per_position_reward

    def create_masked_sequences(
        self, sequences: Union[List[str], pd.DataFrame], num_masked_sites: int
    ) -> Union[List[str], pd.DataFrame]:
        """
        Creates a series of masked sequences from a list or dataframe of unmasked sequences.

        Returns a dataframe with the original sequences and masked sequences in
        the "prompt" column.
        """
        seq_df = sequences
        if isinstance(sequences, list):
            seq_df = pd.DataFrame({"sequence": sequences})

        seq_df["prompt"] = self._create_masked_sequences(
            seq_df["sequence"].tolist(), num_masked_sites
        )
        return seq_df

    @abstractmethod
    def _create_masked_sequences(
        self, sequences: List[str], num_masked_sites: int
    ) -> Union[List[str], pd.DataFrame]:
        """
        Creates a series of masked sequences from a list or dataframe of unmasked sequences.
        Should be implemented by subclasses.
        """
        pass

    def get_position_stats(self):
        """
        Returns a DataFrame containing the statistics of all of the positions
        and the reward values for each position.

        Returns:
            pd.DataFrame: A DataFrame containing the statistics of all of the positions
        """
        df = pd.DataFrame(self.position_stats).rename_axis("position_ind")
        return df

    def get_position_stats_row_form(self):
        """
        Returns the position stat df as a row. Useful for logging to wandb
        """
        df_reshaped = self.get_position_stats().stack().to_frame().T
        df_reshaped.columns = [f"{col}_{idx}" for idx, col in df_reshaped.columns]
        return df_reshaped

    def get_stat(self, name: str, type: str = "sum"):
        """
        Returns a summary statistic of the position statistics of designable
        positions.

        Args:
            name (str): The name of the statistic to return.
            type (str): The type of statistic to return. Options are "sum" or "mean".
                Default is "sum".

        Returns:
            float: The summary statistic of the position statistics of designable positions.
        """
        # Collect all position stats from designable positions
        all_values = [self.position_stats[pos][name] for pos in self.designable_positions]

        if type == "sum":
            return float(np.sum(all_values))
        elif type == "mean":
            return float(np.mean(all_values))
        else:
            raise ValueError(f"Invalid type: {type}")


@MASK_STRATEGY_REGISTRY.register("random")
class RandomMaskingStrategy(MaskingStrategy):
    """
    Implements a random masking strategy that randomly selects positions to mask
    """

    def _create_masked_sequences(
        self, sequences: Union[List[str], pd.DataFrame], num_masked_sites: int
    ):
        """
        Randomly selects num_masked_sites positions to mask for each sequence.
        """
        return [
            mask_k(
                seq,
                k=num_masked_sites,
                mask_str=self.mask_char,
                fixed_indices=self.fixed_position_indices,
            )
            for seq in sequences
        ]


@MASK_STRATEGY_REGISTRY.register("ducb")
class DUCBMaskingStrategy(MaskingStrategy):
    """
    Discounted UCB (D-UCB) for position selection.

    This masker implements the D-UCB multi-armed bandit algorithm where each
    designable residue position is an "arm". It uses discounted sums to handle
    non-stationarity by giving exponentially more weight to recent observations,
    as described in Garivier & Moulines (2008, 2011).
    """

    def __init__(
        self,
        sequence_length: int,
        fixed_position_indices: List[int] = None,
        mask_char: str = "_",
        alpha: float = 2.0,
        gamma: float = 0.95,
    ):
        self.alpha = alpha
        self.gamma = gamma
        super().__init__(sequence_length, fixed_position_indices, mask_char)
        self.initialize(
            position_stat_defaults={
                "discounted_reward_sum": 0,
                "discounted_counts": 0,
            }
        )

    def _update_position_stats(self, masked_positions: int, reward_for_string: float):
        """
        Updates the D-UCB position statistics
        """
        per_position_reward = (
            reward_for_string / len(masked_positions) if len(masked_positions) > 0 else 0.0
        )

        # Apply gamma discount factor to all arms
        for pos in self.designable_positions:
            self.position_stats[pos]["discounted_reward_sum"] *= self.gamma
            self.position_stats[pos]["discounted_counts"] *= self.gamma

        # Add the new observation to the position statistics
        for pos in masked_positions:
            self.position_stats[pos]["discounted_reward_sum"] += per_position_reward
            self.position_stats[pos]["discounted_counts"] += 1.0

    def _create_masked_sequences(self, sequences: List[str], num_masked_sites: int):
        """
        Creates a list of masked sequences from a list of unmasked sequences.
        """

        # Calculate the sum of the discounted counts (handles log(0) for early samples)
        log_sum_dcounts = np.log(max(1, self.get_stat("discounted_counts", type="sum")))

        # Calculate ucb_scores
        ucb_scores = {}
        for pos in self.designable_positions:

            if self.position_stats[pos]["discounted_counts"] == 0:
                # Optimistically initialize
                ucb_scores[pos] = float("inf")
            else:
                # Mean reward estimate
                mean_reward = (
                    self.position_stats[pos]["discounted_reward_sum"]
                    / self.position_stats[pos]["discounted_counts"]
                )

                # UCB exploration term
                exploration_term = self.alpha * np.sqrt(
                    log_sum_dcounts / self.position_stats[pos]["discounted_counts"]
                )

                # Update UCB score
                ucb_scores[pos] = mean_reward + exploration_term

        # Sort the designable positions by their UCB score in descending order
        sorted_positions = sorted(ucb_scores, key=ucb_scores.get, reverse=True)

        # Select the top M positions, ensuring we don't request more than available
        mask_indices = sorted_positions[:num_masked_sites]

        # With this implementation, all sequences in a batch will have the
        # same masked positions.
        return [
            mask_assigned_positions(seq, mask_indices, mask_str=self.mask_char)
            for seq in sequences
        ]


@MASK_STRATEGY_REGISTRY.register("gaussian_thompson")
class GaussianThompsonMaskingStrategy(MaskingStrategy):
    """
    Discounted Thompson Sampling for position selection using a Gaussian model.

    This tracker implements the Discounted Thompson Sampling multi-armed bandit
    algorithm where each designable residue position is an "arm". It uses a
    discount factor (`gamma`) to handle non-stationarity by giving exponentially
    more weight to recent observations. Rewards are modeled using a Gaussian
    (Normal) distribution, which can handle any real-valued reward.
    """

    def __init__(
        self,
        sequence_length: int,
        fixed_position_indices: List[int] = None,
        mask_char: str = "_",
        alpha: float = 2.0,
        gamma: float = 0.95,
    ):
        self.alpha = alpha
        self.gamma = gamma
        super().__init__(sequence_length, fixed_position_indices, mask_char)
        self.initialize(
            position_stat_defaults={
                "discounted_counts": 0,
                "discounted_reward_sum": 0,
                "discounted_reward_square": 0,
            }
        )

    def _update_position_stats(self, masked_positions: int, reward_for_string: float):
        """
        Updates the Gaussian distribution parameters for each position with
        discount factor
        """

        per_position_reward = (
            reward_for_string / len(masked_positions) if len(masked_positions) > 0 else 0.0
        )

        # Apply gamma discount factor to all arms
        for pos in self.designable_positions:
            self.position_stats[pos]["discounted_counts"] *= self.gamma
            self.position_stats[pos]["discounted_reward_sum"] *= self.gamma
            self.position_stats[pos]["discounted_reward_square"] *= self.gamma

        # Add the new observation to the position statistics
        for pos in masked_positions:
            self.position_stats[pos]["discounted_reward_sum"] += per_position_reward
            self.position_stats[pos]["discounted_reward_square"] += per_position_reward**2
            self.position_stats[pos]["discounted_counts"] += 1.0

    def _create_masked_sequences(self, sequences: List[str], num_masked_sites: int):
        """
        Create a list of masked sequences from a list of unmasked sequences.
        """

        # Calculate the mean and variance for each position
        position_distributions = {}
        for pos in self.designable_positions:
            n = max(1, self.position_stats[pos]["discounted_counts"])
            sum_r = self.position_stats[pos]["discounted_reward_sum"]
            sum_r2 = self.position_stats[pos]["discounted_reward_square"]

            # Estimate mean and variance from discounted statistics
            mu_hat = sum_r / n
            var_hat = max((sum_r2 / n) - (mu_hat**2), 1e-9)

            # Sample from the posterior distribution of the mean
            post_std = np.sqrt(var_hat / n)

            position_distributions[pos] = {"mu": mu_hat, "std": post_std}

        masked_sequences = []
        for sequence in sequences:
            # Sample a score for each position
            ts_scores = {}
            for pos in self.designable_positions:
                ts_scores[pos] = np.random.normal(
                    loc=position_distributions[pos]["mu"],
                    scale=position_distributions[pos]["std"]
                )

            # Sort positions by their sampled score in descending order
            sorted_positions = sorted(ts_scores, key=ts_scores.get, reverse=True)

            # Mask the top M positions
            masked_sequences.append(
                mask_assigned_positions(
                    sequence,
                    sorted_positions[:num_masked_sites],
                    mask_str=self.mask_char,
                )
            )

        return masked_sequences
