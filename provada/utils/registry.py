"""
registry.py
"""

from typing import Dict, Any
import inspect


class ComponentClassRegistry:
    """
    A registry for component subclasses.
    """

    def __init__(self):
        """
        Initialize the registry.
        """
        self._components: Dict[str, type] = {}

    def register(self, name=None):
        """
        Decorator to register a component subclass under a given name.
        """

        def decorator(cls):
            """
            Decorator to register a component subclass.
            """
            component_name = name or cls.__name__.lower()
            self._components[component_name] = cls
            return cls

        return decorator

    def get_class(self, name):
        """
        Returns a component class of the given name.
        """
        if name not in self._components:
            raise ValueError(
                f"Component {name} not found in registry. Available components: {list(self._components.keys())}"
            )
        return self._components.get(name)

    def get_instance(self, name, **kwargs):
        """
        Returns a component instance of the given name.
        """
        return self.get_class(name)(**kwargs)

    def get_default_kwargs(self, name: str) -> Dict[str, Any]:
        """
        Returns the default keyword argument values from the __init__ method
        of the specified component class as a dictionary.

        Args:
            name: The name of the registered component class

        Returns:
            Dict[str, Any]: A dictionary mapping parameter names to their default values.
                           Only parameters with default values are included.

        Raises:
            ValueError: If the component name is not found in the registry
        """
        cls = self.get_class(name)

        # Get the signature of the __init__ method
        init_signature = inspect.signature(cls.__init__)

        # Extract parameters with default values (excluding 'self')
        default_kwargs = {}
        for param_name, param in init_signature.parameters.items():
            if param_name != "self" and param.default is not inspect.Parameter.empty:
                default_kwargs[param_name] = param.default

        return default_kwargs

    def list_available_class_names(self):
        """
        Returns a list of names of the available component classes.
        """
        return list(self._components.keys())

    def list_available_classes(self):
        """
        Returns a list of the available component classes.
        """
        return list(self._components.values())

    def __str__(self):
        """
        Returns a string representation of the registry.
        """
        return f"{self.__class__.__name__}({self.list_available_class_names()})"

    def __repr__(self):
        """
        Returns a string representation of the registry.
        """
        return self.__str__()


# Create registries for different component types
GENERATOR_REGISTRY = ComponentClassRegistry()
MASK_STRATEGY_REGISTRY = ComponentClassRegistry()
EVALUATOR_REGISTRY = ComponentClassRegistry()
SCHEDULER_REGISTRY = ComponentClassRegistry()
SAMPLER_REGISTRY = ComponentClassRegistry()
