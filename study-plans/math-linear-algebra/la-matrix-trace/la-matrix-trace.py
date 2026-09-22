import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a Python float.
    """
    trace_of_A = float(sum(map(lambda i: A[i][i], range(len(A)))))
    return trace_of_A