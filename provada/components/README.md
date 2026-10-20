# ProVADA Components

Components are the core building blocks of ProVADA's directed evolution framework. They define how protein sequences are generated, evaluated, and optimized during the adaptive search process. ProVADA comes with several built-in components for common use cases, but the real power lies in creating custom components tailored to your specific protein design objectives.

## Overview

The ProVADA framework uses three main component types that work together during optimization:

```
┌─────────────────┐
│    Masking      │  Masks a fraction of the designable positions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generator     │  Creates new sequences by filling masked positions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluator     │  Scores sequences based on desired properties
└─────────────────┘
```

Masking is not a component: the sampler masks a random fraction of the
designable positions each iteration, and the fraction comes from the masking
schedule. Generators and evaluators are both modular and extensible—you can mix
and match the built-ins or create your own.

## Component Types

### 1. Evaluators

**What they do**: Score protein sequences based on properties you care about (localization, stability, binding affinity, etc.)

**When to customize**: When you have domain-specific objectives or want to use custom machine learning models for scoring

**Learn more**: [EVALUATORS.md](./EVALUATORS.md)

Built-in evaluators include:
- `sequence_similarity` - Hamming distance, identity, and other sequence metrics
- `localization_predictor` - Subcellular localization prediction using ESM2
- `structure` - ESMFold2 structure prediction, with SAP and Rosetta energy
- `Immunogenicity` - Predicted MHC class I presentation from netMHCpan
- `mpnn_score` and `caliby_energy_score` - The generator's own score, carried through
- `discounted_hamming_distance` - Distance to the reference discounted by localization probability
- `dummy` - Counts methionines; the smallest worked example of an evaluator

### 2. Generators

**What they do**: Generate new protein sequences by filling in masked (designable) positions

**When to customize**: When you want to use a different protein language model, implement custom sampling strategies, or incorporate domain knowledge into sequence design

**Learn more**: [GENERATORS.md](./GENERATORS.md)

Built-in generators include:
- `esm2` and `esm3` - Protein language models for masked infilling
- `mpnn`, `soluble_mpnn` and `abmpnn` - ProteinMPNN weight sets, through the JAX toolkit
- `mpnn_pytorch` and `soluble_mpnn_pytorch` - The same weights through the PyTorch toolkit
- `caliby` and `soluble_caliby` - Caliby's Potts model, optionally ensemble-conditioned
- `random` - Random amino acid sampling (baseline)

## Quick Start: Using Built-in Components

To use built-in components, simply specify them in your YAML configuration:

```yaml
# Configure the generator
generator:
  generator_type: esm3
  temperature: 0.7

# Configure evaluators via score weights
sampler:
  score_weights:
    normalized_hamming_distance: 1.0   # Distance from the reference sequence
    localization_prob: 2.0             # Predicted localization

# Configure how much of the sequence is redesigned each iteration
masking_schedule:
  schedule_type: linear
  start_value: 0.2
  stop_value: 0.05
```

ProVADA will automatically discover and instantiate the appropriate components.

## Creating Custom Components

Each component type has a simple interface requiring just 1-2 methods to implement:

**Evaluators**: Implement `available_scores()` and `evaluate_sequences()`
```python
@EVALUATOR_REGISTRY.register("my_score")
class MyEvaluator(Evaluator):
    @classmethod
    def available_scores(cls):
        return {"my_score": {"larger_is_better": True, ...}}

    def evaluate_sequences(self, sequences, **kwargs):
        # Your scoring logic
        return pd.DataFrame({"sequence": sequences, "my_score": scores})
```

**Generators**: Implement `_generate()`
```python
@GENERATOR_REGISTRY.register("my_generator")
class MyGenerator(Generator):
    def _generate(self, prompts, **kwargs):
        # Your generation logic
        return [{"sequence": filled_seq, "prompt": prompt} for ...]
```

## Component Registration

All components use registry-based discovery. When you register a component with a decorator:

```python
@EVALUATOR_REGISTRY.register("my_custom_evaluator")
class MyCustomEvaluator(Evaluator):
    ...
```

ProVADA automatically makes it available throughout the framework. You can then reference it by name in configuration files without any additional imports or setup.

## File Organization

```
provada/components/
├── README.md                    # This file
├── evaluator.py                 # Evaluator base class + built-ins
├── generator.py                 # Generator base class + built-ins
├── EVALUATORS.md               # Detailed evaluator documentation
└── GENERATORS.md               # Detailed generator documentation
```

## Common Patterns

### Using Base Variant Information

Components often need information about the reference protein (structure, sequence, etc.). Override `get_extra_kwargs()` to access it:

```python
def get_extra_kwargs(self, base_variant):
    return {
        "reference_sequence": base_variant.sequence,
        "structure": base_variant.structure
    }
```

### Caching

Evaluators support automatic caching to avoid re-scoring identical sequences. Control it via:

```python
MyEvaluator(use_cache=True)  # Default behavior
```

## Testing Custom Components

Before using a custom component in a full run, test it independently:

```python
# Test an evaluator
evaluator = MyEvaluator()
results = evaluator.evaluate_sequences(["MKTAYIAKQR", "MKTAYIAKQR"])
print(results)

# Test a generator
generator = MyGenerator()
prompts = ["MK__YI_KQR"]
results = generator.generate(prompts)
print(results)
```

## Getting Help

For detailed information on implementing each component type, see:
- [EVALUATORS.md](./EVALUATORS.md) - Complete guide to custom scoring functions
- [GENERATORS.md](./GENERATORS.md) - Complete guide to custom sequence generators

For examples, check the built-in implementations in `evaluator.py` and `generator.py`.
