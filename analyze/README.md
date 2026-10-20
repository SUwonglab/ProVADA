# Provada Analysis Module

This module provides utilities for analyzing and visualizing the results of provada runs.

## Features

- **Data loading**: read a run's trajectory with each sequence's scores merged in
- **Visualization**: scatter plots of the trajectory through score space

## What a run writes

`sequence_lookup.csv` holds the scores of every evaluated state, the reference
and the initial particles included. `load_run_data` joins the trajectory to it
by the retained-state hash, so every row carries the identity and scores of both
the sequence that was generated and the one that was kept.

`final_population.csv` holds the final particles, duplicates included.
Initialization rows carry iteration `-1`.

To reconstruct the particle population at any iteration, repeat each row by the
number of particles that retained it:

```python
rows = load_run_data(result_dir)
rows = rows[rows["iteration"] == iteration]
population = rows.loc[rows.index.repeat(rows["selected_count"])]
```

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

## Module Structure

- `loading.py`: Functions for loading and processing run data
  - `load_run_data()`: Main function to load trajectory with scores
  - `bin_by_iteration()`: Bin trajectory points by iteration ranges
  - `merge_scores_into_trajectory()`: Merge sequence scores into trajectory

- `plotting.py`: Visualization functions
  - `show_trajectory_plot()`: Create static scatter plots with gradients


## Color Gradients

The visualization functions use continuous color gradients (not discrete bins) to show progression through iterations. This provides a smooth, intuitive view of how the optimization progressed.

Recommended colormaps:
- `viridis` (default): Perceptually uniform, colorblind-friendly
- `plasma`: High contrast, good for presentations
- `inferno`: Warm colors, dramatic
- `magma`: Similar to inferno, slightly different hue
- `coolwarm`: Diverging colormap, blue to red
- `RdYlBu`: Red-Yellow-Blue diverging

## Worked example

[`example_notebook.ipynb`](./example_notebook.ipynb) runs through all of this
against a real run: loading, the score columns a run carries, the trajectory
plot, fitness over iterations, and reconstructing a population.

## Tips

- Use `alpha` parameter to control point transparency for dense plots
