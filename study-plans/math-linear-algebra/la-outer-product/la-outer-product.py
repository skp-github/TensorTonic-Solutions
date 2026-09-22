import numpy as np

def outer_product(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 outer-product matrix.
    """
    output = np.array(
        [list(map(lambda x : x * y , v)) for y in u], 
        dtype = np.float64
    )
    return output
            