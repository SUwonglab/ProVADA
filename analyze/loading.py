"""
loading.py

Functions for loading and processing the output of provada runs.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from provada.sequences.io import hash_sequence
from provada.utils.log import get_logger

logger = get_logger(__name__)


def load_run_data(
    result_dir: Union[str, Path],
    drop_duplicates: bool = True,
    num_iteration_bins: Optional[int] = None,
    merge_scores: bool = True,
) -> pd.DataFrame:
    """
    Load and process the complete trajectory data from a provada run directory.

    This function loads both the generated_sequences.csv and trajectory.csv files,
    optionally removes duplicate sequences, bins the trajectory by iteration,
    and merges the sequence scores into the trajectory dataframe.

    Args:
        result_dir: Path to the results directory containing the run outputs
        drop_duplicates: Whether to drop duplicate sequences from generated_sequences.csv.
            Default is True for backwards compatibility with older runs.
        num_iteration_bins: Number of bins to divide the iterations into. If None,
            each iteration gets its own bin. Useful for grouping iterations for
            visualization purposes.
        merge_scores: Whether to merge sequence scores into the trajectory.
            Default is True.

    Returns:
        A merged dataframe containing trajectory data with sequence scores.
        If merge_scores is False, returns the trajectory dataframe with iteration
        binning applied (if requested).

    Raises:
        FileNotFoundError: If the result directory or required CSV files don't exist
        ValueError: If the CSV files are empty or malformed

    Example:
        >>> df = load_run_data("../results/my_run", num_iteration_bins=10)
        >>> # Now df contains trajectory data with scores, binned into 10 iteration groups
    """
    result_dir = Path(result_dir)

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    # Define paths to the required files
    sequences_path = result_dir / "generated_sequences.csv"
    trajectory_path = result_dir / "trajectory.csv"

    # Check that files exist
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    logger.info(f"Loading sequences from {sequences_path}")
    sequences_df = pd.read_csv(sequences_path)

    if sequences_df.empty:
        raise ValueError(f"Sequences file is empty: {sequences_path}")

    # Drop duplicates if requested
    if drop_duplicates:
        initial_count = len(sequences_df)
        sequences_df = sequences_df.drop_duplicates(subset="sequence")
        duplicates_dropped = initial_count - len(sequences_df)
        if duplicates_dropped > 0:
            logger.info(f"Dropped {duplicates_dropped} duplicate sequences")

    logger.info(f"Loading trajectory from {trajectory_path}")
    trajectory_df = pd.read_csv(trajectory_path)

    if trajectory_df.empty:
        raise ValueError(f"Trajectory file is empty: {trajectory_path}")

    # Bin by iteration if requested
    if num_iteration_bins is not None:
        logger.info(f"Binning trajectory into {num_iteration_bins} iteration bins")
        trajectory_df = bin_by_iteration(trajectory_df, num_bins=num_iteration_bins)

    # Merge scores if requested
    if merge_scores:
        logger.info("Merging scores into trajectory")
        trajectory_df = merge_scores_into_trajectory(trajectory_df, sequences_df)

    logger.info(f"Successfully loaded run data: {len(trajectory_df)} trajectory entries")

    return trajectory_df


def bin_by_iteration(
    trajectory_df: pd.DataFrame,
    num_bins: Optional[int] = None,
    iteration_col: str = "iteration",
) -> pd.DataFrame:
    """
    Bin the trajectory instances by iteration ranges.

    This function assigns each row in the trajectory to an iteration bin,
    which is useful for grouping data across iterations for visualization
    or analysis purposes.

    Args:
        trajectory_df: The trajectory dataframe to bin
        num_bins: The number of bins to create. If None, uses the number of
            iterations (i.e., one bin per iteration).
        iteration_col: The name of the column containing iteration numbers.
            Default is "iteration".

    Returns:
        The trajectory dataframe with two additional columns:
            - iteration_bin_range: A categorical column showing the range of
              iterations in each bin (e.g., "(0.0, 10.0]")
            - iteration_bin: An integer column with the bin number (0-indexed)

    Example:
        >>> df = bin_by_iteration(trajectory_df, num_bins=10)
        >>> # Now df has iteration_bin and iteration_bin_range columns
    """
    # Make a copy to avoid modifying the original dataframe
    trajectory_df = trajectory_df.copy()

    # Determine the number of iterations
    num_iterations = trajectory_df[iteration_col].max() + 1  # +1 because we start at 0

    # Determine where the boundaries of the bins should be
    if num_bins is None:
        num_bins = num_iterations

    bin_boundaries = np.linspace(0, num_iterations, num_bins + 1)

    # Assign each instance to a bin
    trajectory_df["iteration_bin_range"] = pd.cut(
        trajectory_df[iteration_col], bins=bin_boundaries, include_lowest=True
    )

    # Number the bin
    trajectory_df["iteration_bin"] = trajectory_df["iteration_bin_range"].cat.codes

    logger.debug(f"Binned {len(trajectory_df)} entries into {num_bins} iteration bins")

    return trajectory_df


def merge_scores_into_trajectory(
    trajectory_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    iteration_col: str = "iteration",
) -> pd.DataFrame:
    """
    Merge sequence scores into the trajectory dataframe.

    This function uses sequence hashing to join the scores from generated_sequences.csv
    into the trajectory.csv data, allowing for analysis of how scores evolved over
    the course of the run.

    Args:
        trajectory_df: The trajectory dataframe containing sequence hashes
        sequences_df: The sequences dataframe containing sequences and their scores
        iteration_col: The name of the iteration column (unused, kept for backwards
            compatibility with notebook code)

    Returns:
        The trajectory dataframe with all columns from sequences_df merged in,
        matched by sequence_hash

    Note:
        - The merge is performed using a left join with many-to-one validation,
          meaning each trajectory entry should map to exactly one sequence
        - The sequence column is dropped from sequences_df before merging to avoid
          redundancy (trajectory already contains sequence_hash)
    """
    # Make copies to avoid modifying the original dataframes
    trajectory_df = trajectory_df.copy()
    sequences_df = sequences_df.copy()

    # Hash all sequences in sequences_df
    logger.debug("Hashing sequences for merge")
    sequences_df["sequence_hash"] = sequences_df["sequence"].apply(hash_sequence)

    # Drop the sequence column to avoid redundancy
    sequences_df = sequences_df.drop(columns=["sequence"])

    # Merge the scores into the trajectory dataframe
    logger.debug(
        f"Merging {len(sequences_df)} unique sequences into trajectory with {len(trajectory_df)} entries"
    )

    trajectory_df = trajectory_df.merge(
        sequences_df, on="sequence_hash", how="left", validate="many_to_one"
    )

    # Check for any rows that didn't get matched
    unmatched = (
        trajectory_df[sequences_df.columns[sequences_df.columns != "sequence_hash"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    if unmatched > 0:
        logger.warning(
            f"{unmatched} trajectory entries did not match any sequences in generated_sequences.csv"
        )

    return trajectory_df
