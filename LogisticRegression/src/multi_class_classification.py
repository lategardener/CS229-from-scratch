import numpy as np


# ==========================================================
# Logit function (linear combination)
# ==========================================================
def logit(x, theta):
    """
    Compute the logit (linear combination) θ^T * x.
    This is the input to the sigmoid in logistic regression,
    or the input to the softmax in multi-class classification.

    Parameters:
    - x: feature vector (shape: [n_features] or [n_features, 1])
    - theta: weights vector (shape: [n_features, 1])

    Returns:
    - linear combination θ^T * x (float)
    """
    return theta.T @ x

# ==========================================================
# Probability for class i (multiclass)
# ==========================================================
def proba_class_i(x, theta_list, i):
    """
    Compute the probability that input x belongs to class i
    using the softmax formula for multi-class logistic regression.

    Parameters:
    - x: feature vector (shape: [n_features, 1] or [n_features])
    - theta_list: list of weight vectors, one per class
                  (theta_list[j] has shape [n_features, 1])
    - i: the class index (1-based, i.e., i=1 for the first class)

    Returns:
    - probability that x belongs to class i (float between 0 and 1)
    """
    # Numerator: exp(logit for class i)
    numerator = np.exp(logit(x, theta_list[i - 1]))

    # Denominator: sum of exp(logit) over all classes
    denominator = np.sum([np.exp(logit(x, theta_j)) for theta_j in theta_list])

    # Softmax probability
    return numerator / denominator