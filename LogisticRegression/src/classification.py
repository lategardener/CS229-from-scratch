import numpy as np


def h_function(x, w):
    return 1 / (1 + np.exp(-(w.T.dot(x))))


def log_likelihood(X, y, w):
    return np.sum([y[i] * np.log(h_function(X[i], w)) + (1 - y[i] * np.log(1 - h_function(X[i], w))) for i in range(len(X))])