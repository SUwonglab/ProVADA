# Masking Strategies: Deciding Which Positions to Design

Masking strategies determine which positions in your protein sequence should be redesigned at each iteration. This is a crucial part of ProVADA's adaptive optimization—instead of redesigning the entire sequence every time, the masking strategy intelligently selects which positions to focus on based on past performance. While ProVADA includes several sophisticated strategies based on multi-armed bandit algorithms, you can create your own to implement different exploration-exploitation trade-offs.

## Understanding Masking Strategies

At each iteration, the masking strategy looks at the current population of sequences and decides which positions to mask (mark for redesign). These masked positions are then filled by the generator to create new sequence variants. The masking strategy learns over time which positions lead to better scores when modified, allowing ProVADA to focus computational effort where it matters most.

The masking strategy maintains statistics about each designable position (positions not marked as fixed) and uses these statistics to make masking decisions. After each iteration, the strategy updates these statistics based on the rewards (scores) of the generated sequences.

## Creating Your Own Masking Strategy

Building a custom masking strategy requires implementing one core method, with optional customization of how the strategy learns from rewards.

### Step 1: Import and Register Your Strategy

Start by importing the base class and registering your strategy with a unique name:

```python
from provada.components.masking import MaskingStrategy
from provada.utils.registry import MASK_STRATEGY_REGISTRY
from provada.sequences.mask import mask_assigned_positions
from typing import List
import numpy as np

@MASK_STRATEGY_REGISTRY.register("my_custom_masking")
class MyCustomMaskingStrategy(MaskingStrategy):
    """
    A custom masking strategy that selects positions using [your approach].
    """
```

The registration name (here `"my_custom_masking"`) is what you'll use in your YAML configuration.

### Step 2: Implement `_create_masked_sequences()`

This is the only required method. It receives a list of sequences and the number of positions to mask, and returns masked versions of those sequences:

```python
    def _create_masked_sequences(
        self,
        sequences: List[str],
        num_masked_sites: int
    ) -> List[str]:
        """
        Creates masked sequences by selecting positions to redesign.

        Args:
            sequences: List of sequences to mask
            num_masked_sites: Number of positions to mask in each sequence

        Returns:
            List of masked sequences (same length as input)
        """
        masked_sequences = []

        for seq in sequences:
            # Your logic to select which positions to mask
            positions_to_mask = self._select_positions(num_masked_sites)

            # Create the masked sequence
            masked_seq = mask_assigned_positions(
                seq,
                positions_to_mask,
                mask_str=self.mask_char
            )
            masked_sequences.append(masked_seq)

        return masked_sequences
```

Key points:
- You have access to `self.designable_positions` (positions not fixed)
- You have access to `self.position_stats` (statistics you're tracking)
- Use `mask_assigned_positions()` to create the actual masked strings
- The mask character is stored in `self.mask_char` (defaults to `"_"`)

### Step 3: Add Custom Parameters (Optional)

You'll likely want custom parameters for your strategy. Add them via the constructor, making sure to call the parent constructor:

```python
    def __init__(
        self,
        sequence_length: int,
        fixed_position_indices: List[int] = None,
        mask_char: str = "_",
        my_exploration_param: float = 1.0  # Your custom parameter
    ):
        """
        Initialize the masking strategy.

        Args:
            sequence_length: Length of the protein sequence
            fixed_position_indices: Positions that should never be masked
            mask_char: Character to use for masking (default "_")
            my_exploration_param: Your custom exploration parameter
        """
        self.my_exploration_param = my_exploration_param
        super().__init__(sequence_length, fixed_position_indices, mask_char)
```

## How Position Statistics Work

The masking strategy automatically maintains statistics for each designable position in `self.position_stats`, which is a dictionary structured like:

```python
{
    0: {"selection_count": 5, "reward": 12.3, ...},
    1: {"selection_count": 3, "reward": 8.1, ...},
    ...
}
```

By default, two statistics are tracked:
- `selection_count`: How many times this position has been masked
- `reward`: Cumulative reward received when this position was masked

After each iteration, the `update()` method is called automatically, which updates these statistics based on which positions were masked and what rewards (scores) were achieved.

## Optional: Custom Statistics Initialization

If you want to track additional statistics beyond the defaults, override the `initialize()` method:

```python
    def __init__(self, sequence_length, fixed_position_indices=None, mask_char="_"):
        super().__init__(sequence_length, fixed_position_indices, mask_char)

        # Initialize with custom statistics
        self.initialize(position_stat_defaults={
            "selection_count": 0,
            "reward": 0,
            "my_custom_stat": 0.0,  # Your additional statistic
        })
```

## Optional: Custom Reward Update Logic

By default, when a sequence receives a reward, that reward is divided equally among all masked positions. You can customize this by overriding `_update_position_stats()`:

```python
    def _update_position_stats(self, masked_positions: List[int], reward_for_string: float):
        """
        Updates position statistics based on which positions were masked and the reward received.

        Args:
            masked_positions: List of position indices that were masked
            reward_for_string: The reward (score) achieved for this sequence
        """
        # Your custom logic for distributing rewards
        for pos in masked_positions:
            self.position_stats[pos]["reward"] += reward_for_string
            # Update other custom statistics
            self.position_stats[pos]["my_custom_stat"] = ...
```

## Built-in Masking Strategies

ProVADA includes several strategies you can use as references:

- **`RandomMaskingStrategy`** (`random`) - Randomly selects positions to mask; simplest baseline
- **`DUCBMaskingStrategy`** (`ducb`) - Discounted UCB algorithm for non-stationary bandits
- **`GaussianThompsonMaskingStrategy`** (`gaussian_thompson`) - Thompson sampling with Gaussian reward model

Check out `provada/components/masking.py` to see their complete implementations.

## Using Your Masking Strategy

Once you've created your strategy, specify it in your YAML configuration:

```yaml
masking_strategy:
  masking_strategy_type: my_custom_masking
  masking_strategy_kwargs:
    my_exploration_param: 2.0
    # Any other parameters
```

ProVADA will:
- Instantiate your strategy at the beginning of the run
- Call it at each iteration to create masked sequences
- Automatically update statistics after each iteration
- Log position statistics to `position_stats.csv`

## Understanding the Masking Schedule

The number of positions to mask (`num_masked_sites`) is controlled by a separate masking schedule, not the masking strategy itself. The schedule typically starts high (explore broadly) and decreases over iterations (exploit good positions).

Your masking strategy receives `num_masked_sites` as an argument and should select exactly that many positions.

## Complete Example: Epsilon-Greedy Strategy

Here's a complete example of a simple epsilon-greedy masking strategy that exploits high-reward positions most of the time but occasionally explores randomly:

```python
from provada.components.masking import MaskingStrategy
from provada.utils.registry import MASK_STRATEGY_REGISTRY
from provada.sequences.mask import mask_assigned_positions
from typing import List
import numpy as np

@MASK_STRATEGY_REGISTRY.register("epsilon_greedy")
class EpsilonGreedyMaskingStrategy(MaskingStrategy):
    """
    Epsilon-greedy position selection: exploit high-reward positions with
    probability (1-epsilon), explore randomly with probability epsilon.
    """

    def __init__(
        self,
        sequence_length: int,
        fixed_position_indices: List[int] = None,
        mask_char: str = "_",
        epsilon: float = 0.1
    ):
        """
        Args:
            epsilon: Probability of random exploration (0-1)
        """
        self.epsilon = epsilon
        super().__init__(sequence_length, fixed_position_indices, mask_char)

    def _create_masked_sequences(
        self,
        sequences: List[str],
        num_masked_sites: int
    ) -> List[str]:
        """
        Select positions using epsilon-greedy approach.
        """
        masked_sequences = []

        for seq in sequences:
            if np.random.random() < self.epsilon:
                # Explore: random positions
                positions_to_mask = np.random.choice(
                    self.designable_positions,
                    size=min(num_masked_sites, len(self.designable_positions)),
                    replace=False
                )
            else:
                # Exploit: select positions with highest average reward
                avg_rewards = {}
                for pos in self.designable_positions:
                    count = self.position_stats[pos]["selection_count"]
                    if count > 0:
                        avg_rewards[pos] = (
                            self.position_stats[pos]["reward"] / count
                        )
                    else:
                        # Optimistically initialize unexplored positions
                        avg_rewards[pos] = float('inf')

                # Sort by average reward and take top K
                sorted_positions = sorted(
                    avg_rewards.keys(),
                    key=avg_rewards.get,
                    reverse=True
                )
                positions_to_mask = sorted_positions[:num_masked_sites]

            # Create masked sequence
            masked_seq = mask_assigned_positions(
                seq,
                positions_to_mask,
                mask_str=self.mask_char
            )
            masked_sequences.append(masked_seq)

        return masked_sequences
```

Use it in your config:

```yaml
masking_strategy:
  masking_strategy_type: epsilon_greedy
  masking_strategy_kwargs:
    epsilon: 0.15
```

## Advanced: Handling Non-Stationary Rewards

If you expect the value of positions to change over time (non-stationarity), consider using discounting like the built-in `DUCBMaskingStrategy`. This gives more weight to recent observations:

```python
    def _update_position_stats(self, masked_positions: List[int], reward_for_string: float):
        """Apply exponential discounting to all positions before updating."""
        gamma = 0.95  # Discount factor

        # Discount all existing statistics
        for pos in self.designable_positions:
            self.position_stats[pos]["reward"] *= gamma
            self.position_stats[pos]["selection_count"] *= gamma

        # Add new observations
        per_position_reward = reward_for_string / len(masked_positions)
        for pos in masked_positions:
            self.position_stats[pos]["reward"] += per_position_reward
            self.position_stats[pos]["selection_count"] += 1.0
```

## Tips and Best Practices

**Start simple**: Begin with a simple strategy (even random) and add complexity only if needed.

**Balance exploration and exploitation**: Good strategies explore enough to find valuable positions but exploit known good positions to improve quickly.

**Handle cold-start**: Decide how to treat positions with no observations (optimistic initialization often works well).

**Consider correlations**: Positions often interact. Advanced strategies might consider which positions are masked together.

**Monitor statistics**: Check `position_stats.csv` to see if your strategy is exploring appropriately.

**Respect fixed positions**: The base class automatically handles `fixed_position_indices`—just select from `self.designable_positions`.

**Match sequence diversity to population size**: If you mask the same positions for all sequences in a batch, you'll get less diversity than if you mask different positions for each sequence.

You now have everything you need to create custom masking strategies that guide ProVADA's adaptive optimization process!
