import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    numerator = np.dot(a, b)

    norm_a = sum(map(lambda x: x**2, a)) ** 0.5
    norm_b = sum(map(lambda x: x**2, b)) ** 0.5

    denominator = norm_a * norm_b

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)