# ProVADA Components

Components are the core building blocks of ProVADA's directed evolution framework. They define how protein sequences are generated, evaluated, and optimized during the adaptive search process. ProVADA comes with several built-in components for common use cases, but the real power lies in creating custom components tailored to your specific protein design objectives.

## Overview

The ProVADA framework uses three main component types that work together during optimization:

```
┌─────────────────┐
│ Masking Strategy│  Decides which positions to redesign
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

Each component type is modular and extensible—you can mix and match built-in components or create your own to implement novel design strategies.

## Component Types

### 1. Evaluators

**What they do**: Score protein sequences based on properties you care about (localization, stability, binding affinity, etc.)

**When to customize**: When you have domain-specific objectives or want to use custom machine learning models for scoring

**Learn more**: [EVALUATORS.md](./EVALUATORS.md)

Built-in evaluators include:
- `sequence_similarity` - Hamming distance, identity, and other sequence metrics
- `localization_predictor` - Subcellular localization prediction using ESM2
- `structure` - Structure prediction and quality metrics (SAP, SASA, Rosetta energy)

### 2. Generators

**What they do**: Generate new protein sequences by filling in masked (designable) positions

**When to customize**: When you want to use a different protein language model, implement custom sampling strategies, or incorporate domain knowledge into sequence design

**Learn more**: [GENERATORS.md](./GENERATORS.md)

Built-in generators include:
- `esm3` - ESM3 protein language model for iterative design
- `mpnn` - ProteinMPNN for structure-conditioned design
- `soluble_mpnn` - ProteinMPNN variant optimized for solubility
- `random` - Random amino acid sampling (baseline)

### 3. Masking Strategies

**What they do**: Decide which positions in the sequence to redesign at each iteration based on learned statistics

**When to customize**: When you want to implement different exploration-exploitation trade-offs or incorporate structural/functional constraints into position selection

**Learn more**: [MASKING.md](./MASKING.md)

Built-in strategies include:
- `random` - Random position selection (baseline)
- `ducb` - Discounted Upper Confidence Bound for non-stationary rewards
- `gaussian_thompson` - Thompson sampling with Gaussian reward models

## Quick Start: Using Built-in Components

To use built-in components, simply specify them in your YAML configuration:

```yaml
# Configure the generator
generator:
  generator_type: esm3
  generation_kwargs:
    temperature: 0.7

# Configure evaluators via score weights
sampler:
  score_weights:
    hamming_distance: -1.0          # Minimize distance from reference
    predicted_mitochondria: 1.0     # Maximize mitochondrial localization

# Configure the masking strategy
masking_strategy:
  masking_strategy_type: ducb
  masking_strategy_kwargs:
    gamma: 0.95
    zeta: 1.0
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

**Masking Strategies**: Implement `_create_masked_sequences()`
```python
@MASK_STRATEGY_REGISTRY.register("my_masking")
class MyMaskingStrategy(MaskingStrategy):
    def _create_masked_sequences(self, sequences, num_masked_sites):
        # Your position selection logic
        return [masked_seq for ...]
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
├── masking.py                   # Masking strategy base class + built-ins
├── EVALUATORS.md               # Detailed evaluator documentation
├── GENERATORS.md               # Detailed generator documentation
└── MASKING.md                  # Detailed masking documentation
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

### Worker Process Initialization

For components that use heavy models or libraries, initialize them once per worker process:

```python
@staticmethod
def worker_init():
    model = load_heavy_model()
    return {"model": model, "success": True}
```

Then access in your component methods:

```python
from provada.utils.multiprocess import _worker_local
ctx = getattr(_worker_local, "ctx", {})
model = ctx["model"]
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

# Test a masking strategy
strategy = MyMaskingStrategy(sequence_length=10)
masked = strategy.create_masked_sequences(["MKTAYIAKQR"], num_masked_sites=3)
print(masked)
```

## Getting Help

For detailed information on implementing each component type, see:
- [EVALUATORS.md](./EVALUATORS.md) - Complete guide to custom scoring functions
- [GENERATORS.md](./GENERATORS.md) - Complete guide to custom sequence generators
- [MASKING.md](./MASKING.md) - Complete guide to custom masking strategies

For examples, check the built-in implementations in `evaluator.py`, `generator.py`, and `masking.py`.
