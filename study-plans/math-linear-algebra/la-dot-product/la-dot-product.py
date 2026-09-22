import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    output = sum(map(lambda a, b : a*b, x, y))
    return float(output)