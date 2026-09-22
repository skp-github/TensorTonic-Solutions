import numpy as np

def linear_combination(vectors: list, coefficients: list) -> np.ndarray:
    """
    Returns the weighted sum as a float64 vector.
    """
    return np.array(list(map(lambda i: sum(map(lambda v, c: v[i] * c, vectors, coefficients)), range(len(vectors[0])))), dtype=np.float64)