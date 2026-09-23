import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    n_features = len(X[0])
    W = np.zeros(n_features)
    b = 0.0
    X = np.array(X)
    Y = np.array(y)
    for _ in range(epochs):
        for x, y in zip(X, Y):
            z = np.dot(W, x) + b
            predictions = 1 if z >=0 else 0
            error = y - predictions
            W += lr * error * x
            b += lr * error 

    return W.tolist(), b

    
