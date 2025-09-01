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
    numerator = np.exp(logit(x, theta_list[i]))

    # Denominator: sum of exp(logit) over all classes
    denominator = np.sum([np.exp(logit(x, theta_j)) for theta_j in theta_list])

    # Softmax probability
    return numerator / denominator


def cross_entropy(x, theta_list, class_x):
    return - np.log(proba_class_i(x, theta_list, class_x))

def negative_log_likelihood(X, y, theta_list):
    return sum([cross_entropy(X[i], theta_list, y[i].item()) for i in range(len(X))])


def gradient_descent(
        X, y,
        epsilon=0.001, max_iter=1000,
        type="BGD", alpha=1e-4, batch_size=32,
):

    # Initialize weights to zeros
    classes = np.unique(y)
    nb_classes =  len(classes)
    theta_list = [np.zeros((X.shape[1], 1)) for _ in range(nb_classes)]

    # Compute initial cost
    cost = negative_log_likelihood(X, y, theta_list)
    loss_history = [cost]             # track loss evolution

    # ----------------------
    # Batch GD and Mini-Batch GD
    # ----------------------
    if type in ["BGD", "MBGD"]:
        for _ in range(max_iter):
            for i in classes:
                if type == "BGD":
                    # Full dataset gradient
                    grad = np.sum([(proba_class_i(X[j], theta_list, i) - 1 * (y[j]==i).item()) * X[j].reshape(-1, 1) for j in range(len(X))]) / len(X)
                else:
                    # Mini-batch gradient
                    chosen_data = np.random.choice(len(X), batch_size, replace=False)
                    grad = np.sum([(proba_class_i(X[j], theta_list, i) - 1 * (y[j]==i).item()) * X[j].reshape(-1, 1)  for j in chosen_data]) / batch_size

                # Weight update
                theta_list[i] = theta_list[i] - alpha * grad                # print(theta_list)

            # Update cost and track progress
            new_cost = negative_log_likelihood(X, y, theta_list)
            loss_history.append(new_cost)

            # Convergence check
            if abs(new_cost - cost) < epsilon:
                print("Convergence achieved!")
                break
            else:
                cost = new_cost

    # ----------------------
    # Stochastic Gradient Descent
    # ----------------------
    elif type == "SGD":
        for j in range(len(X)):
            for i in classes:
                grad = (proba_class_i(X[j], theta_list, i) - 1 * (y[j]==i).item()) * X[j].reshape(-1, 1)
                theta_list[i] = theta_list[i] - alpha * grad

            # Track after each update
            cost = negative_log_likelihood(X, y, theta_list)
            loss_history.append(cost)

    else:
        raise ValueError("Invalid type. Choose 'BGD', 'SGD', or 'MBGD'.")

    # Final results
    final_loss = loss_history[-1]
    loss_history = np.array(loss_history, dtype=np.float64)

    return theta_list, final_loss, loss_history