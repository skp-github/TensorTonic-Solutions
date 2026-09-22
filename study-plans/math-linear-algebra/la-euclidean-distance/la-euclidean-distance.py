import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    output = float(sum(map(lambda a, b : (a - b)**2, x, y )))**.5
    return output