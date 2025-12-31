"""
plotting.py

Functions for visualizing provada run trajectories and results.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from matplotlib.figure import Figure
from tqdm import tqdm

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


def create_trajectory_animation(
    trajectory_df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Union[str, Path],
    result_name: Optional[str] = None,
    hue: str = "iteration",
    cmap: str = "viridis",
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (10, 8),
    fps: int = 30,
    points_per_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    title_template: Optional[str] = None,
    dpi: int = 100,
    show_final_frames: int = 15,
    linear_distribution: bool = True,
    **kwargs,
) -> Path:
    """
    Create an animated MOV video showing the trajectory being built point by point.

    This function creates an animation that reveals the trajectory progressively,
    with points colored by a continuous gradient (typically iteration number).

    Animation parameters are automatically calculated based on dataset size to ensure
    reasonable generation time (~1-2 minutes). For 100k points, this creates ~80 frames.

    Output files are saved to analyze/output/{result_name}/ for organized storage.

    Args:
        trajectory_df: The trajectory dataframe to animate
        x: Column name for x-axis
        y: Column name for y-axis
        output_path: Path where the MOV video should be saved (filename only, or
            full path if result_name is None)
        result_name: Name of the result folder for organizing outputs. If provided,
            output will be saved to analyze/output/{result_name}/{output_path}.
            If None, uses output_path as-is.
        hue: Column name for coloring points. Default is "iteration".
            Should be a numeric column for gradient coloring.
        cmap: Colormap name for the gradient. Default is "viridis".
        alpha: Transparency of points (0-1). Default is 0.6.
        figsize: Figure size as (width, height). Default is (10, 8).
        fps: Frames per second for the animation. Default is 30.
        points_per_frame: Number of points to add per frame. If None (default),
            automatically calculated based on total points to keep generation
            time reasonable. Higher values make the animation faster.
        max_frames: Maximum number of animation frames (excluding final hold).
            If None (default), automatically determined based on dataset size:
            - <= 1k points: 50 frames
            - <= 10k points: 60 frames
            - <= 50k points: 70 frames
            - > 50k points: 80 frames
        title_template: Template for the title. Can include {iteration} and {total}
            placeholders. If None, uses a default template.
        dpi: DPI for the output video. Default is 100.
        show_final_frames: Number of extra frames to show the complete plot
            at the end. Default is 15 (0.5 seconds at 30fps).
        linear_distribution: If True, frames are evenly distributed across
            iterations. If False (default), more frames are allocated to earlier iterations
            where trajectories typically change more rapidly.
        **kwargs: Additional arguments passed to plt.scatter

    Returns:
        Path to the created MOV file

    Example:
        >>> # Auto-calculated parameters (recommended for large datasets)
        >>> create_trajectory_animation(
        ...     df,  # 100k points
        ...     x="localization_prob",
        ...     y="sequence_similarity",
        ...     output_path="trajectory.mov",
        ...     result_name="my_run_2024_01_15",
        ... )
        >>> # Creates ~80 frames automatically (~1-2 min generation time)
        >>> # Saved to analyze/output/my_run_2024_01_15/trajectory.mov

        >>> # Use nonlinear distribution for smoother early trajectory
        >>> create_trajectory_animation(
        ...     df,
        ...     x="localization_prob",
        ...     y="sequence_similarity",
        ...     output_path="trajectory.mov",
        ...     result_name="my_run",
        ...     linear_distribution=False,  # More frames for early iterations
        ... )

        >>> # Manual control for specific animation speed
        >>> create_trajectory_animation(
        ...     df,
        ...     x="localization_prob",
        ...     y="sequence_similarity",
        ...     output_path="trajectory.mov",
        ...     result_name="my_run",
        ...     points_per_frame=100,  # Add 100 points per frame
        ... )

    Note:
        This function requires ffmpeg for MOV video creation.
        The dataframe is sorted by the hue column before animation.
        A progress bar is displayed during rendering.
    """
    # Handle output path with result_name
    if result_name is not None:
        # Create output directory structure: analyze/output/{result_name}/
        script_dir = Path(__file__).parent  # This is the analyze/ directory
        output_dir = script_dir / "output" / result_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(output_path).name
    else:
        output_path = Path(output_path)

    # Validate columns
    for col in [x, y, hue]:
        if col not in trajectory_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    # Sort by hue column to show progression
    logger.info(f"Sorting trajectory by {hue} for animation")
    trajectory_df = trajectory_df.sort_values(hue).reset_index(drop=True)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get data ranges for consistent axes
    x_data = trajectory_df[x]
    y_data = trajectory_df[y]
    hue_data = trajectory_df[hue]

    x_margin = (x_data.max() - x_data.min()) * 0.05
    y_margin = (y_data.max() - y_data.min()) * 0.05

    ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
    ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

    # Set labels
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())

    # Create normalization for color mapping
    norm = Normalize(vmin=hue_data.min(), vmax=hue_data.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(hue.replace("_", " ").title(), rotation=270, labelpad=20)

    # Default title template
    if title_template is None:
        title_template = "Trajectory Animation - Point {iteration} / {total}"

    # Initialize scatter plot
    scatter = ax.scatter([], [], c=[], cmap=cmap, alpha=alpha, norm=norm, **kwargs)

    # Total number of points
    total_points = len(trajectory_df)

    # Auto-calculate points_per_frame and max_frames if not specified
    if points_per_frame is None and max_frames is None:
        # Auto-determine based on dataset size
        # Targets for ~1-2 minute generation time
        if total_points <= 1000:
            max_frames = 50  # Small datasets
        elif total_points <= 10000:
            max_frames = 60  # Medium datasets
        elif total_points <= 50000:
            max_frames = 70  # Larger datasets
        else:
            max_frames = 80  # Very large datasets

        points_per_frame = max(1, total_points // max_frames)
        logger.info(
            f"Auto-calculated animation parameters: {points_per_frame} points/frame "
            f"(target {max_frames} frames for {total_points} points)"
        )
    elif points_per_frame is None:
        # max_frames specified, calculate points_per_frame
        points_per_frame = max(1, total_points // max_frames)
        logger.info(
            f"Calculated {points_per_frame} points/frame to achieve ~{max_frames} frames"
        )
    elif max_frames is None:
        # points_per_frame specified, use it as-is
        logger.info(f"Using specified {points_per_frame} points/frame")
    else:
        # Both specified - warn if they conflict
        calculated_frames = total_points // points_per_frame
        if calculated_frames > max_frames * 1.5:
            logger.warning(
                f"Specified points_per_frame={points_per_frame} will create "
                f"~{calculated_frames} frames, exceeding max_frames={max_frames}. "
                f"Consider increasing points_per_frame for faster rendering."
            )

    num_animation_frames = int(np.ceil(total_points / points_per_frame))

    logger.info(
        f"Creating animation with {num_animation_frames} frames "
        f"({total_points} points, {points_per_frame} points/frame)"
    )

    # Pre-calculate point indices for each frame if using nonlinear distribution
    if not linear_distribution:
        # Use power function to allocate more frames to early iterations
        # power < 1 creates a concave curve: slow growth early, faster later
        # This means more frames show early trajectory changes
        power = 0.6  # Adjust this to control distribution (lower = more early frames)
        frame_to_points = []
        for frame_idx in range(num_animation_frames):
            # Normalized frame position (0 to 1)
            norm_frame = (frame_idx + 1) / num_animation_frames
            # Apply power function to get nonlinear progression
            norm_points = norm_frame**power
            points_to_show = int(norm_points * total_points)
            frame_to_points.append(min(points_to_show, total_points))
        logger.info(
            f"Using nonlinear frame distribution (power={power}) for smoother early trajectory"
        )
    else:
        frame_to_points = None

    def init():
        """Initialize the animation."""
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        return (scatter,)

    def update(frame):
        """Update function for each frame."""
        # Calculate how many points to show
        if frame_to_points is not None:
            # Use pre-calculated nonlinear mapping
            points_to_show = frame_to_points[min(frame, len(frame_to_points) - 1)]
        else:
            # Linear distribution
            points_to_show = min((frame + 1) * points_per_frame, total_points)

        # Get data up to this point
        x_vals = x_data.iloc[:points_to_show].values
        y_vals = y_data.iloc[:points_to_show].values
        colors = hue_data.iloc[:points_to_show].values

        # Update scatter plot
        scatter.set_offsets(np.column_stack([x_vals, y_vals]))
        scatter.set_array(colors)

        # Update title
        title = title_template.format(iteration=points_to_show, total=total_points)
        ax.set_title(title)

        return (scatter,)

    # Create animation
    total_frames = num_animation_frames + show_final_frames
    logger.info(
        f"Total frames (including {show_final_frames} final hold frames): {total_frames}"
    )

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=total_frames, interval=1000 / fps, blit=True
    )

    # Save as MOV with progress bar
    logger.info(f"Saving animation to {output_path}")

    # Create a custom progress callback
    pbar = tqdm(total=total_frames, desc="Rendering frames", unit="frame")

    # Use FFMpegWriter with mpeg4 codec (more universally available than libx264)
    from matplotlib.animation import FFMpegWriter

    writer = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)

    # Monkey-patch the writer to add progress updates
    original_grab_frame = writer.grab_frame
    frame_count = [0]  # Use list to allow modification in nested function

    def grab_frame_with_progress(*args, **kwargs):
        result = original_grab_frame(*args, **kwargs)
        frame_count[0] += 1
        pbar.update(1)
        return result

    writer.grab_frame = grab_frame_with_progress

    anim.save(output_path, writer=writer, dpi=dpi)
    pbar.close()

    plt.close(fig)

    logger.info(f"Animation saved successfully to {output_path}")

    return output_path


def create_multi_trajectory_animation(
    trajectory_df: pd.DataFrame,
    plot_configs: List[Tuple[str, str]],
    output_path: Union[str, Path],
    result_name: Optional[str] = None,
    hue: str = "iteration",
    cmap: str = "viridis",
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (15, 5),
    fps: int = 30,
    points_per_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    dpi: int = 100,
    show_final_frames: int = 15,
    linear_distribution: bool = False,
    **kwargs,
) -> Path:
    """
    Create an animated MOV video with multiple trajectory plots side by side.

    Output files are saved to analyze/output/{result_name}/ for organized storage.

    Args:
        trajectory_df: The trajectory dataframe to animate
        plot_configs: List of (x, y) column pairs for each subplot.
            Example: [("score1", "score2"), ("score1", "score3")]
        output_path: Path where the MOV video should be saved (filename only, or
            full path if result_name is None)
        result_name: Name of the result folder for organizing outputs. If provided,
            output will be saved to analyze/output/{result_name}/{output_path}.
            If None, uses output_path as-is.
        hue: Column name for coloring points. Default is "iteration".
        cmap: Colormap name for the gradient. Default is "viridis".
        alpha: Transparency of points (0-1). Default is 0.6.
        figsize: Figure size as (width, height). Default is (15, 5).
        fps: Frames per second for the animation. Default is 30.
        points_per_frame: Number of points to add per frame. If None, auto-calculated.
        max_frames: Maximum number of animation frames. If None, auto-calculated
            based on dataset size (50-80 frames depending on size).
        dpi: DPI for the output video. Default is 100.
        show_final_frames: Number of extra frames to show at the end. Default is 15.
        linear_distribution: If True (default), frames are evenly distributed across
            iterations. If False, more frames are allocated to earlier iterations.
        **kwargs: Additional arguments passed to plt.scatter

    Returns:
        Path to the created MOV file

    Example:
        >>> create_multi_trajectory_animation(
        ...     df,
        ...     plot_configs=[
        ...         ("localization_prob", "sequence_similarity"),
        ...         ("localization_prob", "sap_score_normalized"),
        ...     ],
        ...     output_path="multi_trajectory.mov",
        ...     result_name="my_run_2024_01_15",
        ... )
    """
    # Handle output path with result_name
    if result_name is not None:
        # Create output directory structure: analyze/output/{result_name}/
        script_dir = Path(__file__).parent  # This is the analyze/ directory
        output_dir = script_dir / "output" / result_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(output_path).name
    else:
        output_path = Path(output_path)

    # Validate columns
    for x, y in plot_configs:
        for col in [x, y]:
            if col not in trajectory_df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")

    if hue not in trajectory_df.columns:
        raise ValueError(f"Hue column '{hue}' not found in dataframe")

    # Sort by hue column
    logger.info(f"Sorting trajectory by {hue} for animation")
    trajectory_df = trajectory_df.sort_values(hue).reset_index(drop=True)

    # Set up the figure with subplots
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    # Ensure axes is always a list
    if num_plots == 1:
        axes = [axes]

    # Create normalization for color mapping
    hue_data = trajectory_df[hue]
    norm = Normalize(vmin=hue_data.min(), vmax=hue_data.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Initialize each subplot
    scatters = []
    for idx, (ax, (x, y)) in enumerate(zip(axes, plot_configs)):
        x_data = trajectory_df[x]
        y_data = trajectory_df[y]

        # Set axis limits
        x_margin = (x_data.max() - x_data.min()) * 0.05
        y_margin = (y_data.max() - y_data.min()) * 0.05
        ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
        ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

        # Set labels
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(y.replace("_", " ").title())

        # Initialize scatter
        scatter = ax.scatter([], [], c=[], cmap=cmap, alpha=alpha, norm=norm, **kwargs)
        scatters.append((scatter, x, y))

    # Add single colorbar to the figure
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label(hue.replace("_", " ").title(), rotation=270, labelpad=20)

    # Animation setup
    total_points = len(trajectory_df)

    # Auto-calculate points_per_frame and max_frames if not specified (same logic as single plot)
    if points_per_frame is None and max_frames is None:
        # Targets for ~1-2 minute generation time
        if total_points <= 1000:
            max_frames = 50
        elif total_points <= 10000:
            max_frames = 60
        elif total_points <= 50000:
            max_frames = 70
        else:
            max_frames = 80
        points_per_frame = max(1, total_points // max_frames)
        logger.info(
            f"Auto-calculated animation parameters: {points_per_frame} points/frame "
            f"(target {max_frames} frames for {total_points} points)"
        )
    elif points_per_frame is None:
        points_per_frame = max(1, total_points // max_frames)
        logger.info(
            f"Calculated {points_per_frame} points/frame to achieve ~{max_frames} frames"
        )
    elif max_frames is None:
        logger.info(f"Using specified {points_per_frame} points/frame")
    else:
        calculated_frames = total_points // points_per_frame
        if calculated_frames > max_frames * 1.5:
            logger.warning(
                f"Specified points_per_frame={points_per_frame} will create "
                f"~{calculated_frames} frames, exceeding max_frames={max_frames}"
            )

    num_animation_frames = int(np.ceil(total_points / points_per_frame))

    logger.info(
        f"Creating multi-plot animation with {num_animation_frames} frames "
        f"({total_points} points, {points_per_frame} points/frame, {num_plots} subplots)"
    )

    # Pre-calculate point indices for each frame if using nonlinear distribution
    if not linear_distribution:
        power = 0.6
        frame_to_points = []
        for frame_idx in range(num_animation_frames):
            norm_frame = (frame_idx + 1) / num_animation_frames
            norm_points = norm_frame**power
            points_to_show = int(norm_points * total_points)
            frame_to_points.append(min(points_to_show, total_points))
        logger.info(
            f"Using nonlinear frame distribution (power={power}) for smoother early trajectory"
        )
    else:
        frame_to_points = None

    def init():
        """Initialize the animation."""
        for scatter, _, _ in scatters:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
        return [s[0] for s in scatters]

    def update(frame):
        """Update function for each frame."""
        if frame_to_points is not None:
            points_to_show = frame_to_points[min(frame, len(frame_to_points) - 1)]
        else:
            points_to_show = min((frame + 1) * points_per_frame, total_points)
        colors = hue_data.iloc[:points_to_show].values

        # Update each subplot
        for scatter, x, y in scatters:
            x_vals = trajectory_df[x].iloc[:points_to_show].values
            y_vals = trajectory_df[y].iloc[:points_to_show].values
            scatter.set_offsets(np.column_stack([x_vals, y_vals]))
            scatter.set_array(colors)

        # Update main title
        fig.suptitle(f"Trajectory Animation - Point {points_to_show} / {total_points}")

        return [s[0] for s in scatters]

    # Create animation
    total_frames = num_animation_frames + show_final_frames
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=total_frames, interval=1000 / fps, blit=True
    )

    # Save as MOV with progress bar
    logger.info(f"Saving multi-plot animation to {output_path}")

    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Rendering frames", unit="frame")

    # Use FFMpegWriter with mpeg4 codec (more universally available than libx264)
    from matplotlib.animation import FFMpegWriter

    writer = FFMpegWriter(fps=fps, codec="mpeg4", bitrate=5000)

    # Monkey-patch the writer to add progress updates
    original_grab_frame = writer.grab_frame

    def grab_frame_with_progress(*args, **kwargs):
        result = original_grab_frame(*args, **kwargs)
        pbar.update(1)
        return result

    writer.grab_frame = grab_frame_with_progress

    anim.save(output_path, writer=writer, dpi=dpi)
    pbar.close()

    plt.close(fig)

    logger.info(f"Multi-plot animation saved successfully to {output_path}")

    return output_path
