import numpy as np

def matrix_multiply(A: list, B: list) -> np.ndarray:
    """
    Returns the matrix product as a float64 array.
    """
    C = [[0] * len(B[0]) for _ in range(len(A))]
    temp_sum = 0
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                temp_sum += A[i][k] * B[k][j]
            C[i][j] = temp_sum
            temp_sum = 0

    return np.array(C, dtype=np.float64)

            