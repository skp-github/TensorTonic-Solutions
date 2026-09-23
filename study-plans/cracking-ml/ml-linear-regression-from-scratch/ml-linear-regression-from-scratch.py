import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    n_features = len(X[0])

    weights = np.zeros(n_features)
    bias = 0 

    for _ in range(epochs):

        y_pred = X @ weights + bias 
        errors = y_pred - y 

        dw = (2/n_samples) * (X.T@errors)
        db = (2/n_samples) * np.sum(errors)

        weights -= lr*dw
        bias -= lr*db


    return weights.tolist(), bias
        

    
