import numpy as np

def matrix_vector_multiply(A: list, x: list) -> np.ndarray:
    """
    Returns the matrix-vector product as a float64 array.
    """
    result = np.array([sum(map(lambda a, xdash: a * xdash, row, x)) for row in A], dtype=np.float64)
    return result