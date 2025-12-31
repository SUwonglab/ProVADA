"""
schedules.py

Implements various parameter schedules for the sampler.
"""

from abc import ABC, abstractmethod
import numpy as np
from provada.utils.registry import SCHEDULER_REGISTRY

def get_schedule(schedule_type: str, **kwargs) -> "ParameterSchedule":
    """
    Returns a schedule of the given type.
    """
    return SCHEDULER_REGISTRY.get_instance(schedule_type, **kwargs)


class ParameterSchedule(ABC):
    """
    Abstract base class for parameter schedules.
    """

    def __init__(self, value: float, total_iters: int):
        """
        Sets up the parameter schedule.
        """
        self.value = value
        self.total_iters = total_iters

    @abstractmethod
    def __call__(self, iteration: int) -> float:
        """
        Returns the value of the parameter at the given iteration.
        """
        NotImplementedError

    def get_schedule_as_array(self) -> np.ndarray:
        """
        Returns the schedule as an array of values.
        """
        return np.array([self(i) for i in range(self.total_iters)])


@SCHEDULER_REGISTRY.register("constant")
class Constant(ParameterSchedule):
    """
    Constant schedule for a parameter.
    """

    def __call__(self, iteration: int) -> float:
        return self.value


@SCHEDULER_REGISTRY.register("linear")
class Linear(ParameterSchedule):
    """
    Linear schedule for a parameter.
    """

    def __init__(self, total_iters: int, start_value: float, stop_value: float):
        """
        Sets up the linear schedule.
        """
        self.total_iters = total_iters
        self.start_value = start_value
        self.stop_value = stop_value

    def __call__(self, iteration: int) -> float:
        """
        Returns the value of the parameter at the given iteration.
        """
        frac = iteration / (self.total_iters - 1)
        return self.start_value + (self.stop_value - self.start_value) * frac


@SCHEDULER_REGISTRY.register("geometric")
class Geometric(ParameterSchedule):
    """
    Geometric schedule for a parameter using np.geomspace.
    Creates a geometrically spaced sequence between start and stop values.
    """

    def __init__(self, total_iters: int, start_value: float, stop_value: float):
        """
        Sets up the geometric schedule.

        Args:
            total_iters (int): The total number of iterations
            start_value (float): Value at iteration 0
            stop_value (float): Value at iteration total_iters-1
        """
        self.total_iters = total_iters
        self.start_value = start_value
        self.stop_value = stop_value

        # Pre-compute the entire schedule using np.geomspace
        if total_iters > 1:
            self.schedule = np.geomspace(
                start_value, stop_value, num=total_iters, endpoint=True
            )
        else:
            self.schedule = np.array([start_value])

    def __call__(self, iteration: int) -> float:
        """
        Returns the value of the parameter at the given iteration.
        """
        if iteration >= self.total_iters:
            return self.stop_value

        return float(self.schedule[iteration])

    def get_schedule_as_array(self) -> np.ndarray:
        """
        Returns the schedule as an array of values.
        """
        return self.schedule.copy()


@SCHEDULER_REGISTRY.register("power")
class PowerLaw(ParameterSchedule):
    """
    Power law schedule for a parameter.
    """

    def __init__(
        self, total_iters: int, start_value: float, stop_value: float, alpha: float
    ):
        """
        Sets up the power law schedule.

        Args:
            total_iters (int): The total number of iterations
            start_value (float): Value at iteration 0
            stop_value (float): Value at iteration total_iters
            alpha (float): Power law exponent
        """
        self.total_iters = total_iters
        self.start_value = start_value
        self.stop_value = stop_value
        self.alpha = alpha

    def __call__(self, iteration: int) -> float:
        """
        Returns the value of the parameter at the given iteration.
        """

        frac = iteration / (self.total_iters - 1)

        return self.start_value + (self.stop_value - self.start_value) * (
            frac**self.alpha
        )
