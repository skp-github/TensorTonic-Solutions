import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transpose as a float64 array.
    """
    rows = len(A)
    cols = len(A[0])
    B = []
    for c in range(cols):
        temp_row = list(map(lambda x: x[c], A))
        B.append(temp_row)

    return np.array(B, dtype=np.float64)
    