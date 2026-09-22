import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    L1_norm = sum(map(lambda a : np.abs(a), v))
    L2_norm = sum(map(lambda a: a*a, v))**.5

    Linf = max(map(lambda a: np.abs(a), v))

    return np.array([L1_norm, L2_norm, Linf])