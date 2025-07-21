# provada/fitness_score/__init__.py
"""
provada.fitness_score subpackage: functions and classes for scoring protein variants.
"""

from .eval_fitness import eval_fitness_score, eval_fitness_scores

__all__ = [
    "eval_fitness_score",
    "eval_fitness_scores"
]