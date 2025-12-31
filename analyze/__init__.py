"""
Provada Analysis Module

This module provides utilities for analyzing the results of provada runs,
including loading trajectory data, merging scores, and visualizing results.
"""

from analyze.loading import (
    load_run_data,
    bin_by_iteration,
    merge_scores_into_trajectory,
)

from analyze.plotting import (
    show_trajectory_plot,
    create_trajectory_animation,
    create_multi_trajectory_animation,
)

__all__ = [
    "load_run_data",
    "bin_by_iteration",
    "merge_scores_into_trajectory",
    "show_trajectory_plot",
    "create_trajectory_animation",
    "create_multi_trajectory_animation",
]
