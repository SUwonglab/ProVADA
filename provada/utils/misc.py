"""
misc.py

Miscellaneous utility functions.
"""

def set_nested_attr(obj, attr_path, value):
    """
    Set a nested attribute in an object using a dot notation path.

    Args:
        obj: The object to set the nested attribute in.
        attr_path: The dot notation path to the nested attribute.
        value: The value to set the nested attribute to.
    """
    attributes = attr_path.split(".")
    for attr in attributes[:-1]:  # Go through all but the last attribute
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)


def get_nested_attr(obj, attr_path):
    """
    Get a nested attribute from an object using a dot notation path.

    Args:
        obj: The object to get the nested attribute from.
        attr_path: The dot notation path to the nested attribute.

    Returns:
        The nested attribute.
    """
    attributes = attr_path.split(".")
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj
