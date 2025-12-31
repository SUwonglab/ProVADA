# Generators: Creating Custom Sequence Generators

Generators are the creative engine of ProVADA—they take partially masked protein sequences and fill in the masked positions to create new sequence variants. While ProVADA includes generators for ESM3 and ProteinMPNN, you can create your own generator to leverage different models, sampling strategies, or design approaches.

## Understanding Generators

A generator receives "prompts" (sequences with masked positions represented as `_` characters) and returns complete sequences where the masked positions have been filled in. The generator is called repeatedly during the sampling loop, creating diverse sequence variants that are then evaluated and selected based on their scores. Generators handle the core sequence design logic while ProVADA manages the iteration, selection, and optimization process.

## Creating Your Own Generator

Building a custom generator involves creating a class that inherits from `Generator` and implementing a single required method. Let's walk through it step by step.

### Step 1: Import and Register Your Generator

Start by importing the base class and registering your generator with a unique name:

```python
from provada.components.generator import Generator
from provada.utils.registry import GENERATOR_REGISTRY
from typing import List
import pandas as pd

@GENERATOR_REGISTRY.register("my_custom_generator")
class MyCustomGenerator(Generator):
    """
    A custom generator that fills masked positions using [your approach].
    """
```

The registration name (here `"my_custom_generator"`) is what you'll use in your YAML configuration to specify this generator.

### Step 2: Implement `_generate()`

This is the only required method. It receives a list of masked sequences (prompts) and returns a list of dictionaries containing the generated sequences:

```python
    def _generate(self, prompts: List[str], **kwargs) -> List[dict]:
        """
        Generates complete sequences from masked prompts.

        Args:
            prompts: List of sequences with masked positions (indicated by '_')
            **kwargs: Additional generation parameters

        Returns:
            List of dicts, each containing at minimum:
                - "sequence": The complete generated sequence
                - "prompt": The original masked prompt
        """
        generations = []

        for prompt in prompts:
            # Your generation logic here
            generated_seq = self._fill_masked_positions(prompt)

            generations.append({
                "sequence": generated_seq,
                "prompt": prompt
            })

        return generations
```

The key points:
- **Input**: Each prompt is a string where `_` marks positions to be designed
- **Output**: Each dict must have `"sequence"` (the filled-in sequence) and `"prompt"` (the original)
- You can add extra metadata to the output dicts (e.g., model scores, confidence values)

### Step 3: Add Custom Parameters (Optional)

You'll typically want to initialize your generator with parameters like model paths, temperature, etc. Add these via the constructor:

```python
    def __init__(self, seed: int = 42, temperature: float = 1.0):
        """
        Initialize your generator with custom parameters.

        Args:
            seed: Random seed for reproducibility
            temperature: Sampling temperature (or any custom param)
        """
        self.seed = seed
        self.temperature = temperature

        # Initialize your model/sampler here
        self.model = self._load_model()
```

These parameters can be passed via the YAML configuration file (see Usage section below).

## Optional: Providing Extra Arguments

If your generator needs information from the base variant (like protein structure for ProteinMPNN), override the `get_extra_kwargs()` method:

```python
    def get_extra_kwargs(self, base_variant):
        """
        Provides additional arguments needed for generation.
        """
        return {
            "structure": base_variant.structure,
            # Add any other required information
        }
```

These kwargs will be automatically merged with your generation parameters and passed to the `generate()` method.

## How Masked Sequences Work

Generators receive sequences where designable positions are marked with `_`. For example:

```
Original:  MKTAYIAKQR
Masked:    MK__YI_KQR
```

Your generator should:
1. Identify positions marked with `_`
2. Fill them with amino acids according to your generation strategy
3. Keep all other positions unchanged (unless you have a specific reason to modify them)

## Built-in Generators

ProVADA includes several generators you can use as references:

- **`ESM3Generator`** (`esm3`) - Uses Meta's ESM3 protein language model for iterative sequence design
- **`MPNNGenerator`** (`mpnn`) - Uses ProteinMPNN for structure-conditioned design
- **`SolubleMPNNGenerator`** (`soluble_mpnn`) - ProteinMPNN variant fine-tuned for solubility
- **`RandomGenerator`** (`random`) - Randomly samples amino acids based on codon schemes

Check out `provada/components/generator.py` to see their full implementations.

## Using Your Generator

Once you've created your generator, specify it in your YAML configuration:

```yaml
generator:
  generator_type: my_custom_generator
  generation_kwargs:
    temperature: 0.8
    # Any other parameters your generator accepts
```

ProVADA will:
- Instantiate your generator with the provided kwargs
- Call it at each iteration with masked sequences from the population
- Use the generated sequences for evaluation and selection

## Advanced: Multiple Generations Per Prompt

Some generators (like ProteinMPNN) can produce multiple sequence variants from a single prompt. If you want this behavior, you can return multiple entries per prompt:

```python
    def _generate(self, prompts: List[str], generations_per_prompt: int = 1, **kwargs):
        all_generations = []

        for prompt in prompts:
            # Generate multiple variants
            for _ in range(generations_per_prompt):
                generated_seq = self._sample_sequence(prompt)
                all_generations.append({
                    "sequence": generated_seq,
                    "prompt": prompt
                })

        return all_generations
```

Then configure it with:

```yaml
generator:
  generator_type: my_custom_generator
  generation_kwargs:
    generations_per_prompt: 10
```

## Advanced: Batched Generation

For efficiency, you may want to batch your generations:

```python
    def _generate(self, prompts: List[str], batch_size: int = 32, **kwargs):
        generations = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # Generate sequences for the batch
            batch_outputs = self._batch_generate(batch)

            # Collect results
            for prompt, output in zip(batch, batch_outputs):
                generations.append({
                    "sequence": output,
                    "prompt": prompt
                })

        return generations
```

## Complete Example: Random Hydrophobic Generator

Here's a complete example of a generator that preferentially samples hydrophobic amino acids:

```python
from provada.components.generator import Generator
from provada.utils.registry import GENERATOR_REGISTRY
from typing import List
import random

@GENERATOR_REGISTRY.register("hydrophobic_random")
class HydrophobicRandomGenerator(Generator):
    """
    Fills masked positions with random amino acids, biased toward hydrophobic residues.
    """

    HYDROPHOBIC_AAS = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
    HYDROPHILIC_AAS = ['S', 'T', 'N', 'Q', 'C', 'Y', 'K', 'R', 'H', 'D', 'E', 'G']

    def __init__(self, seed: int = 42, hydrophobic_bias: float = 0.7):
        """
        Args:
            seed: Random seed
            hydrophobic_bias: Probability of sampling a hydrophobic AA (0-1)
        """
        self.seed = seed
        self.hydrophobic_bias = hydrophobic_bias
        random.seed(seed)

    def _generate(self, prompts: List[str], **kwargs) -> List[dict]:
        generations = []

        for prompt in prompts:
            # Convert to list for easy manipulation
            seq_list = list(prompt)

            # Fill each masked position
            for i, char in enumerate(seq_list):
                if char == '_':
                    if random.random() < self.hydrophobic_bias:
                        seq_list[i] = random.choice(self.HYDROPHOBIC_AAS)
                    else:
                        seq_list[i] = random.choice(self.HYDROPHILIC_AAS)

            generated_seq = ''.join(seq_list)

            generations.append({
                "sequence": generated_seq,
                "prompt": prompt
            })

        return generations
```

Use it in your config:

```yaml
generator:
  generator_type: hydrophobic_random
  generation_kwargs:
    hydrophobic_bias: 0.8
```

## Tips and Best Practices

**Respect the mask**: Only modify positions marked with `_` unless you have a specific reason to change fixed positions.

**Be deterministic when possible**: If you're using randomness, make sure to set and use the seed for reproducibility.

**Handle edge cases**: Make sure your generator works with sequences that have no masked positions, or only one masked position.

**Add metadata**: Include useful information in your output dicts like model confidence scores, which can be logged and analyzed later.

**Think about speed**: Generators are called many times. Batch operations when possible, and consider GPU utilization if using neural models.

**Test independently**: Test your generator outside the full ProVADA pipeline first to make sure it's filling masks correctly.

That's all you need to create custom generators! The framework handles the rest—calling your generator at each iteration, managing the population, and optimizing toward your objectives.
