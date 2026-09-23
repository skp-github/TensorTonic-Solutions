import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    n_samples = len(X)
    n_features = len(X[0])
    X = np.array(X)
    weights = np.zeros(n_features)
    bias = 0.0 

    for _ in range(n_iters):
        linear = X @ weights + bias
        predictions = 1 / (1 + np.exp(-linear))

        dw = (X.T@(predictions-y) )/ n_samples
        db = np.mean(predictions-y)

        weights -= lr * dw
        bias -= lr * db

    return weights.tolist(), bias