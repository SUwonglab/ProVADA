# Provada Analysis Module

This module provides utilities for analyzing and visualizing the results of provada runs.

## Features

- **Data Loading**: Load and merge trajectory data with sequence scores
- **Static Visualization**: Create beautiful scatter plots with gradient coloring
- **Animated Visualization**: Generate MOV videos showing trajectory evolution over time

### Loading Data

```python
from analyze import load_run_data

# Load trajectory data with scores merged in
df = load_run_data("../results/my_run_directory")

# Load with 10 iteration bins
df = load_run_data("../results/my_run_directory", num_iteration_bins=10)
```

### Static Visualization

```python
from analyze import show_trajectory_plot
import matplotlib.pyplot as plt

# Create a scatter plot with gradient coloring by iteration
fig = show_trajectory_plot(
    df,
    x="localization_prob",
    y="sequence_similarity",
    use_gradient=True,  # Use continuous color gradient
    cmap="viridis",      # Colormap (try: plasma, inferno, magma, coolwarm)
)
plt.show()
```

### Animated Videos

```python
from analyze import create_trajectory_animation

# Create an animated MOV video showing trajectory evolution
# Output is saved to analyze/output/{result_name}/trajectory.mov
create_trajectory_animation(
    df,
    x="localization_prob",
    y="sequence_similarity",
    output_path="trajectory.mov",
    result_name="my_run_2024_01_15",  # Organizes outputs by run name
    fps=30,                            # Frames per second
    cmap="plasma",                     # Colormap
    # Auto-calculated parameters for ~1-2 minute generation time
)

# Use nonlinear distribution to focus more frames on early trajectory
create_trajectory_animation(
    df,
    x="localization_prob",
    y="sequence_similarity",
    output_path="trajectory_early_focus.mov",
    result_name="my_run",
    linear_distribution=False,  # More frames for early iterations
)
```

### Multi-Plot Animations

```python
from analyze import create_multi_trajectory_animation

# Create side-by-side animations
# Output is saved to analyze/output/{result_name}/multi_trajectory.mov
create_multi_trajectory_animation(
    df,
    plot_configs=[
        ("score1", "score2"),
        ("score1", "score3"),
        ("score2", "score3"),
    ],
    output_path="multi_trajectory.mov",
    result_name="my_run_2024_01_15",
    fps=30,
)
```

## Module Structure

- `loading.py`: Functions for loading and processing run data
  - `load_run_data()`: Main function to load trajectory with scores
  - `bin_by_iteration()`: Bin trajectory points by iteration ranges
  - `merge_scores_into_trajectory()`: Merge sequence scores into trajectory

- `plotting.py`: Visualization functions
  - `show_trajectory_plot()`: Create static scatter plots with gradients
  - `create_trajectory_animation()`: Generate animated MOV videos
  - `create_multi_trajectory_animation()`: Generate multi-plot MOV videos


## Color Gradients

The visualization functions use continuous color gradients (not discrete bins) to show progression through iterations. This provides a smooth, intuitive view of how the optimization progressed.

Recommended colormaps:
- `viridis` (default): Perceptually uniform, colorblind-friendly
- `plasma`: High contrast, good for presentations
- `inferno`: Warm colors, dramatic
- `magma`: Similar to inferno, slightly different hue
- `coolwarm`: Diverging colormap, blue to red
- `RdYlBu`: Red-Yellow-Blue diverging

## Tips

- Use `result_name` parameter to organize outputs in `analyze/output/{result_name}/` folders
- Use `linear_distribution=False` to allocate more frames to early iterations (when trajectories change most)
- Use `points_per_frame` to manually control animation speed (higher = faster)
- Use `max_frames` to set a specific frame count target (default: 50-80 based on dataset size)
- Use `fps` to control playback smoothness (30 is standard)
- Use `show_final_frames` to hold the final complete plot
- Use `alpha` parameter to control point transparency for dense plots
- The dataframe is automatically sorted by iteration for animations
- Animation parameters are auto-calculated for ~1-2 minute generation time