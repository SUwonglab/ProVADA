"""
plotting.py

Functions for visualizing provada run trajectories and results.
"""

from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from provada.utils.log import get_logger

logger = get_logger(__name__)


def show_trajectory_plot(
    trajectory_df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = "iteration",
    use_gradient: bool = True,
    cmap: str = "viridis",
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    **kwargs,
) -> Figure:
    """
    Display a scatter plot of the trajectory with optional color gradient.

    Args:
        trajectory_df: The trajectory dataframe to plot
        x: Column name for x-axis
        y: Column name for y-axis
        hue: Column name for coloring points. Default is "iteration".
            If use_gradient is True, this should be a numeric column.
        use_gradient: If True, use a continuous color gradient. If False,
            use discrete colors (categorical). Default is True.
        cmap: Colormap name for the gradient. Default is "viridis".
            Other good options: "plasma", "inferno", "magma", "coolwarm", "RdYlBu"
        alpha: Transparency of points (0-1). Default is 0.6.
        figsize: Figure size as (width, height). Default is (10, 8).
        title: Optional title for the plot
        **kwargs: Additional arguments passed to sns.scatterplot or plt.scatter

    Returns:
        The matplotlib Figure object

    Example:
        >>> fig = show_trajectory_plot(
        ...     df,
        ...     x="localization_prob",
        ...     y="sequence_similarity",
        ...     use_gradient=True,
        ...     cmap="viridis"
        ... )
        >>> plt.show()
    """
    # Validate columns exist
    for col in [x, y]:
        if col not in trajectory_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    if hue and hue not in trajectory_df.columns:
        raise ValueError(f"Hue column '{hue}' not found in dataframe")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if use_gradient and hue:
        # Use continuous color gradient
        hue_values = trajectory_df[hue]

        # Create normalization for color mapping
        norm = Normalize(vmin=hue_values.min(), vmax=hue_values.max())
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # Create scatter plot with gradient
        _ = ax.scatter(
            trajectory_df[x],
            trajectory_df[y],
            c=hue_values,
            cmap=cmap,
            alpha=alpha,
            norm=norm,
            **kwargs,
        )

        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(hue.replace("_", " ").title(), rotation=270, labelpad=20)

    else:
        # Use seaborn for categorical or no hue
        sns.scatterplot(data=trajectory_df, x=x, y=y, hue=hue, alpha=alpha, ax=ax, **kwargs)

    # Set labels
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())

    if title:
        ax.set_title(title)
    else:
        hue_str = f" (colored by {hue})" if hue else ""
        ax.set_title(f"Trajectory: {y} vs {x}{hue_str}")

    plt.tight_layout()

    return fig

