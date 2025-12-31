# Evaluators: Creating Custom Scoring Functions

Evaluators are the scoring functions that assess the quality of generated protein sequences. ProVADA comes with several built-in evaluators for localization prediction, structure quality, sequence similarity, and more, but you can easily create your own custom evaluators to score sequences according to your specific objectives.

## Understanding Evaluators

An evaluator takes a list of protein sequences and returns numerical scores that quantify some property of interest. These scores are then used by the sampler to guide the directed evolution process toward sequences with desired characteristics. Evaluators automatically handle caching, parallel processing, and integration with the sampling loop—you just need to define how to score sequences.

## Creating Your Own Evaluator

To create a custom evaluator, you'll write a new class that inherits from the `Evaluator` base class and implements two required methods. Let's walk through the process.

### Step 1: Import and Set Up Your Class

Start by importing the necessary components and registering your evaluator with a unique name:

```python
from provada.components.evaluator import Evaluator
from provada.utils.registry import EVALUATOR_REGISTRY
import pandas as pd
from typing import List, Dict, Any

@EVALUATOR_REGISTRY.register("my_custom_score")
class MyCustomEvaluator(Evaluator):
    """
    A custom evaluator that scores sequences based on [your criteria].
    """
```

The name you provide to `@EVALUATOR_REGISTRY.register()` is what you'll use in your YAML configuration files to reference this evaluator.

### Step 2: Implement `available_scores()`

Every evaluator must declare what scores it can compute. This class method returns a dictionary describing each score:

```python
    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        """
        Declares the scores this evaluator can compute.
        """
        return {
            "my_score_name": {
                "larger_is_better": True,      # or False if lower is better
                "min_value": 0.0,              # minimum possible value (or None if unknown)
                "max_value": 1.0,              # maximum possible value (or None if unknown)
            }
        }
```

The metadata you provide here helps ProVADA normalize scores and understand optimization direction. If you don't know the exact min/max values, you can set them to `None`, and ProVADA will estimate them from observed data during the run.

You can also use special strings to reference properties of the base variant:

```python
"max_value": "sequence_length"  # Will use the length of the protein sequence
```

### Step 3: Implement `evaluate_sequences()`

This is where you define the actual scoring logic. The method receives a list of sequences and should return a DataFrame with one column per score:

```python
    def evaluate_sequences(self, sequences: List[str], **kwargs) -> pd.DataFrame:
        """
        Scores each sequence according to your custom logic.

        Args:
            sequences: List of protein sequences to score
            **kwargs: Additional arguments (see get_extra_kwargs)

        Returns:
            DataFrame with 'sequence' column and one column per score
        """
        scores = []
        for seq in sequences:
            # Your custom scoring logic here
            score_value = self._compute_score(seq)
            scores.append(score_value)

        return pd.DataFrame({
            "sequence": sequences,
            "my_score_name": scores
        })
```

That's it! These two methods are all you need for a basic evaluator.

## Optional: Adding Extra Arguments

If your evaluator needs information from the base variant (like a reference sequence or structure), override the `get_extra_kwargs()` method:

```python
    def get_extra_kwargs(self, base_variant):
        """
        Provides additional arguments needed for evaluation.
        """
        return {
            "reference_sequence": base_variant.sequence,
            "reference_structure": base_variant.structure
        }
```

These extra kwargs will be automatically passed to your `evaluate_sequences()` method, so you can use them like:

```python
    def evaluate_sequences(
        self,
        sequences: List[str],
        reference_sequence: str = None,
        **kwargs
    ) -> pd.DataFrame:
        # Now you can use reference_sequence
        ...
```

## Optional: Worker Process Initialization

If your evaluator needs to load heavy resources (models, libraries) that should be initialized once per worker process rather than in the main process, override the static `worker_init()` method:

```python
    @staticmethod
    def worker_init():
        """
        Initialize resources in each worker process.
        Returns a dict that gets stored in worker context.
        """
        import some_heavy_library
        model = some_heavy_library.load_model()

        return {
            "model": model,
            "success": True  # Required flag
        }
```

Then access these resources in `evaluate_sequences()`:

```python
    def evaluate_sequences(self, sequences: List[str], **kwargs) -> pd.DataFrame:
        from provada.utils.multiprocess import _worker_local

        # Access worker-local resources
        ctx = getattr(_worker_local, "ctx", {})
        model = ctx["model"]

        # Use the model
        predictions = model.predict(sequences)
        ...
```

## Optional: Customizing Constructor

You can add custom initialization arguments to your evaluator's `__init__()`. Just make sure to call the parent constructor:

```python
    def __init__(
        self,
        seed: int = 42,
        active_scores: List[str] = None,
        use_cache: bool = True,
        my_custom_param: float = 1.0  # Your custom parameter
    ):
        super().__init__(seed=seed, active_scores=active_scores, use_cache=use_cache)
        self.my_custom_param = my_custom_param
```

Then you can pass this parameter in your YAML config via the evaluator kwargs (see Configuration section below).

## Using Your Evaluator

Once you've created your evaluator, use it in your configuration by specifying the score in `score_weights`:

```yaml
sampler:
  score_weights:
    my_score_name: 1.0  # ProVADA will automatically find and use your evaluator
```

ProVADA will automatically:
- Discover that `my_score_name` is provided by your `MyCustomEvaluator`
- Instantiate the evaluator with caching enabled
- Call it during the sampling loop to score sequences
- Cache results to avoid redundant computation

## Built-in Evaluators

ProVADA includes several evaluators you can use as references or extend:

- **`Dummy`** (`dummy`) - Simple example that counts methionines; great starting point
- **`SequenceSimilarity`** (`sequence_similarity`) - Computes metrics like hamming distance, identity, etc.
- **`LocalizationPredictor`** (`localization_predictor`) - Predicts protein localization using ESM2 embeddings
- **`DiscountedHammingDistance`** (`discounted_hamming_distance`) - Combines multiple evaluators into one
- **`StructureMetrics`** (`structure`) - Predicts structures with ESM3 and scores with SAP, SASA, Rosetta energy

Browse `provada/components/evaluator.py` to see these implementations.

## Tips and Best Practices

**Keep scoring fast**: The evaluator will be called many times during a run. If your scoring function is slow, consider batching or using worker processes efficiently.

**Return consistent shapes**: Always return a DataFrame with the same columns (the scores you declared in `available_scores()`).

**Use caching wisely**: The default caching (`use_cache=True`) prevents re-scoring identical sequences. Disable it only if your scores are non-deterministic or depend on iteration state.

**Leverage existing evaluators**: If your score is a combination of existing scores, you can call other evaluators from within yours (see `DiscountedHammingDistance` for an example).

**Handle edge cases**: Make sure your evaluator gracefully handles unusual inputs like very short sequences or sequences with non-standard amino acids.

## Complete Example

Here's a complete minimal evaluator that scores sequences based on their hydrophobicity:

```python
from provada.components.evaluator import Evaluator
from provada.utils.registry import EVALUATOR_REGISTRY
import pandas as pd
from typing import List, Dict, Any

@EVALUATOR_REGISTRY.register("hydrophobicity")
class HydrophobicityEvaluator(Evaluator):
    """
    Scores sequences based on average hydrophobicity using Kyte-Doolittle scale.
    """

    # Kyte-Doolittle hydrophobicity scale
    HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }

    @classmethod
    def available_scores(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "avg_hydrophobicity": {
                "larger_is_better": True,  # Depends on your objective
                "min_value": -4.5,
                "max_value": 4.5,
            }
        }

    def evaluate_sequences(self, sequences: List[str], **kwargs) -> pd.DataFrame:
        scores = []
        for seq in sequences:
            hydro_values = [self.HYDROPHOBICITY.get(aa, 0) for aa in seq]
            avg_hydro = sum(hydro_values) / len(hydro_values) if hydro_values else 0
            scores.append(avg_hydro)

        return pd.DataFrame({
            "sequence": sequences,
            "avg_hydrophobicity": scores
        })
```

Use it in your config:

```yaml
sampler:
  score_weights:
    avg_hydrophobicity: 1.0
```

That's all there is to it! You now have a custom evaluator integrated into the ProVADA framework.
