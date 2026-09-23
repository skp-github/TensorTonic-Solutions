import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_samples, n_features = X.shape

    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    for _ in range(epochs):
        # Predictions
        predictions = X @ weights + bias

        # Errors
        errors = predictions - y

        # Gradients
        dw = (2 / n_samples) * (X.T @ errors) + 2 * alpha * weights
        db = 2 * np.mean(errors)

        # Gradient descent
        weights -= lr * dw
        bias -= lr * db

    return np.round(weights, 4).tolist(), round(bias, 4)