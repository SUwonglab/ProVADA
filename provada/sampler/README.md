# Sampler Module

This directory contains the core sampling algorithms and tracking functionality for ProVADA.

## Weights & Biases Logging

The sampler automatically logs comprehensive metrics to [Weights & Biases](https://wandb.ai/) at each iteration for experiment tracking and visualization. This logging is implemented in `tracking.py`.

### Configuration

To enable wandb logging, specify a `wandb_project` name in your YAML configuration file:

```yaml
wandb_project: my_protein_design_project
```

### Metrics Logged Per Iteration

#### 1. Score Statistics

For each score defined in `score_weights` (e.g., `localization_prob`, `sap_score`, `sequence_similarity`, `rosetta_energy_score`):

- `{score}.iteration_mean` - Mean value across the current population
- `{score}.iteration_min` - Minimum value in the current population
- `{score}.iteration_max` - Maximum value in the current population
- `{score}.all_time_min` - Minimum value observed across all iterations so far
- `{score}.all_time_max` - Maximum value observed across all iterations so far

Additionally, the combined weighted score `SCORE` (computed from all `score_weights`) is logged with the same statistics.

**Example**: If your config has:
```yaml
score_weights:
  localization_prob: 1.0
  sap_score: 2.0
  sequence_similarity: 0.1
```

You'll see metrics like:
- `localization_prob.iteration_mean`
- `localization_prob.all_time_max`
- `sap_score.iteration_mean`
- `SCORE.iteration_mean`
- etc.

#### 2. Run Statistics

- `num_unique_sequences` - Total number of unique sequences generated across all iterations
- `total_sequences_evaluated` - Total sequences evaluated across all iterations, including duplicates

#### 3. Iteration Info

- `step` - Current iteration number (used as the x-axis in wandb plots)

#### 4. Schedule Values

Dynamic schedule values that change over the course of the run:

- `mh_temperature` - Current Metropolis-Hastings temperature from the temperature schedule
- `percent_masked` - Current percentage of positions being masked (from masking schedule)
- `num_masked_sites` - Actual number of positions being masked (calculated as `percent_masked * num_designable_positions`)

#### 5. Acceptance Metrics

- `fraction_mh_acceptances` - Fraction of sequences accepted via the Metropolis-Hastings acceptance criterion in the current iteration

### Additional Wandb Summary Statistics

At the end of the run, the following are saved to `wandb.summary`:

- `active_scores_dict` - Dictionary containing score metadata including min/max value ranges for normalization

### Output Files

In addition to wandb logging, the sampler saves the following CSV files to the `results/{run_name}/` directory:

- **`generated_sequences.csv`** - All unique sequences generated, with their scores (one row per unique sequence)
- **`trajectory.csv`** - Selection history showing which sequences were selected at each iteration
- **`position_stats.csv`** - Per-position statistics from the masking strategy across iterations

### Implementation Details

The wandb logging is implemented via the `TrackingMixin` class in `tracking.py`, which provides:

- `_wandb_log(iteration_df)` - Logs metrics for the current iteration
- `_cache_generations(iteration_df)` - Caches sequences to CSV files
- `cache_trajectory(iteration_df)` - Main entry point that calls both logging and caching methods

### Example Wandb Dashboard

Once logged, you can visualize:
- **Score progression** over iterations (e.g., how `localization_prob.iteration_mean` changes)
- **Schedule dynamics** (e.g., how `mh_temperature` anneals over time)
- **Acceptance rates** (e.g., `fraction_mh_acceptances`)
- **Exploration statistics** (e.g., `num_unique_sequences` growth)

### Disabling Wandb Logging

To run without wandb logging, simply omit the `wandb_project` field from your configuration file. The sampler will still save local CSV files.
