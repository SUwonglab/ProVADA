# Sampler Module

This directory contains the core sampling algorithms and tracking functionality for ProVADA.

## Sampling policy

Both policies run under `sampler_type: mada` and live in the `MADA` class, with
`greedy` selecting between them. Greedy top-K is enabled by default; set
`sampler.greedy: false` (or `--sampler.greedy false` on the CLI) for stochastic sampling.

**Stochastic sampling** is selected with `greedy: false`. It draws the initial population from fully masked generator
completions while retaining fixed residues, then applies Metropolis-Hastings moves
and one resampling pass over all particles. Here `K = N = population_size`, so the
weighted selections become the next population directly, duplicates included.
Stochastic sampling always uses `K = N`; `top_k_fraction` is ignored. If the
configured fraction would give a different K, the sampler logs this once:
`top_k_fraction is ignored for stochastic sampling; using K=N=<population_size>.`

Incremental annealing weights are `exp((1/T_current - 1/T_previous) * starting_SCORE)`,
computed from the parent scores. A parent's weight is shared equally among its
descendants before resampling.

```yaml
sampler:
  sampler_type: mada
  greedy: false
  resampling_method: systematic  # or multinomial
  score_weights:
    localization_prob: 1.0
```

`resampling_method` selects systematic resampling (the default) or independent
weighted draws with replacement; `--sampler.resampling_method multinomial` sets it
on the CLI. Both use the sampler's seeded RNG and preserve the population size.

**Greedy top-K** is the default: `greedy: true`. It starts from the reference
protein, ranks the post-MH rows, retains `K = ceil(top_k_fraction * M)` of the `M`
candidate rows, and resamples those elites back to `population_size`.
`top_k_fraction` defaults to `0.2` in greedy mode. Duplicate
rows count separately: 100 parents producing 10 designs each give `M = 1000`, so a
fraction of 0.2 retains 200 rows. The `rejection` sampler is also available.

## Fitness

`sampler.normalize: true` (the default) scales each metric with bounds fixed for
the whole run, clips it to `[0,1]`, and orients it so higher is better. `SCORE` is
the weighted mean of those components, also in `[0,1]`, and `SCORE_delta` is the
signed improvement over the reference. With `sampler.normalize: false`, `SCORE`
combines oriented raw metrics on their own scales, which generally calls for
different temperatures. Score weights are finite and nonnegative, with at least
one positive weight.

Bounds come from evaluator metadata, or from a reference distribution determined
once before initialization, and hold for the whole run. Bounds given as reference
attributes, such as sequence length, resolve without generating calibration
sequences.

MPNN fitness comes from a fixed-seed scoring pass against the backbone, so
references and candidates are scored the same way, and is cached by sequence,
backbone, chain order and scoring settings. Proposal-time MPNN scores are kept
separately as the `generation_mpnn_score` diagnostic.

## Chains and constraints

For structure-conditioned runs, `/`-separated sequence and prompt segments follow
the structure's chain order. Multirecord FASTA headers identify chains, for
example `>B` and `>A`, and are reordered to the structure's order at load time.
Generated outputs preserve every unmasked residue and the chain layout, and a
whole-chain fixed constraint covers unresolved residues in the supplied sequence.

MPNN scores the reference sequence it is given, even where that differs from the
backbone's native sequence. Caliby scores the structure it is given and requires a
matching reference sequence.

## Output files

Each run writes to `results/{run_name}/`:

- **`generated_sequences.csv`** - every unique sequence generated, with its scores
- **`sequence_lookup.csv`** - scores for every evaluated state, including the reference and the initial particles
- **`trajectory.csv`** - parent, generated-child and retained-state hashes and scores, with acceptance and resampling counts. Initialization is iteration `-1`
- **`final_population.csv`** - the final particles in population order, duplicates kept as separate particle IDs

## Weights & Biases logging

Set `wandb_project` in the config to log a run; omit it to write only the local
CSVs. Each iteration logs:

- `{score}.iteration_mean`, `.iteration_min`, `.iteration_max`, `.all_time_min`
  and `.all_time_max` for every score in `score_weights`, and for the composite
  `SCORE`
- `num_unique_sequences` and `total_sequences_evaluated`
- `mh_temperature`, `percent_masked` and `num_masked_sites` from the schedules
- `fraction_mh_acceptances`
- `step`, the iteration number, as the x-axis

Iteration statistics are taken over the retained population, weighted by how many
particles retained each sequence. At the end of a run, `active_scores_dict` is
written to the W&B summary with each score's normalization bounds.
