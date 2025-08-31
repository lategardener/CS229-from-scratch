import numpy as np
import pandas as pd


# ===========================
# Hypothesis function (sigmoid)
# ===========================
def h_function(x, theta):
    """
    Compute the sigmoid of the linear combination w^T * x.

    Parameters:
    - x: feature vector
    - w: weights vector

    Returns:
    - sigmoid activation (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-(theta.T @ x)))


# ===========================
# Log-likelihood function
# ===========================
def log_likelihood(X, y, theta):
    """
    Compute the log-likelihood of the data under the logistic regression model.

    Parameters:
    - X: feature matrix
    - y: target vector
    - w: weights vector

    Returns:
    - log-likelihood value (the higher, the better)
    """
    return np.sum([
        y[i] * np.log(h_function(X[i], theta)) +
        (1 - y[i]) * np.log(1 - h_function(X[i], theta))
        for i in range(len(X))
    ])


def negative_log_likelihood(X, y, theta):
    return -log_likelihood(X, y, theta)


# ===========================
# Error for a single data point
# ===========================
def error(X, y, theta, i):
    """
    Compute the prediction error for the i-th data point.

    Parameters:
    - X: feature matrix
    - y: target vector
    - theta: weights vector
    - i: index of the data point

    Returns:
    - difference between actual label and predicted probability
    """
    return  y[i] - h_function(X[i], theta)


# ===========================
# Gradient Descent Algorithm
# Supports: BGD, SGD, MBGD
# ===========================
def gradient_descent_ascent(X, y, epsilon=0.001, max_iter=1000, type="BGD", alpha=1e-4, batch_size=32, loss_function=negative_log_likelihood):
    """
    Perform gradient descent to optimize the weights of a logistic regression model.

    Parameters:
    - X: feature matrix (including intercept term)
    - y: target vector
    - epsilon: convergence threshold (stopping criterion based on cost difference)
    - max_iter: maximum number of iterations
    - type: "BGD" (Batch Gradient Descent), "SGD" (Stochastic GD), or "MBGD" (Mini-Batch GD)
    - alpha: learning rate
    - batch_size: size of mini-batches (only relevant if type="MBGD")

    Returns:
    - theta: final weights vector
    - final_loss: last cost value after training
    - weights_history: array of all weight vectors during training
    - loss_history: array of all cost values during training
    """
    # Initialize weights to zeros
    theta = np.zeros((X.shape[1], 1))

    # Compute initial cost
    cost = loss_function(X, y, theta)
    weights_history = [theta.copy()]  # store initial weights
    loss_history = [cost]         # store initial cost


    if loss_function is not negative_log_likelihood and loss_function is not log_likelihood:
        raise ValueError("The specified loss_function is not correct: use negative_log_likelihood or log_likelihood")

    # ----------------------
    # Batch Gradient Descent or Mini-Batch Gradient Descent
    # ----------------------
    if type == "BGD" or type == "MBGD":
        for _ in range(max_iter):
            for j in range(len(theta)):
                if type == "BGD":
                    # Compute gradient over the full dataset
                    grad = np.sum([error(X, y, theta, i) * X[i][j] for i in range(len(X))]) / len(X)
                else:
                    # Compute gradient over a random mini-batch
                    chosen_data = np.random.choice(len(X), batch_size, replace=False)
                    grad = np.sum([error(X, y, theta, i) * X[i][j] for i in chosen_data]) / batch_size

                # Update weight with gradient step
                theta[j] = theta[j] + alpha * grad

            # Store updated weights and cost
            weights_history.append(theta.copy())
            new_cost = loss_function(X, y, theta)
            loss_history.append(new_cost)

            # Check convergence criterion
            difference = abs(new_cost - cost)
            if difference < epsilon:
                print("Convergence achieved!")
                break
            else:
                cost = new_cost

    # ----------------------
    # Stochastic Gradient Descent
    # ----------------------
    elif type == "SGD":
        for i in range(len(X)):
            for j in range(len(theta)):
                grad = error(X, y, theta, i) * X[i][j]
                # Update weight with gradient step
                theta[j] = theta[j] + alpha * grad

            # Store weights and cost after each update
            weights_history.append(theta.copy())
            cost = loss_function(X, y, theta)
            loss_history.append(cost)

    else:
        raise ValueError("Invalid gradient descent type. Choose 'BGD', 'SGD', or 'MBGD'.")

    # Final results
    final_loss = loss_history[-1]
    weights_history = np.array([w.flatten() for w in weights_history])
    loss_history = np.array(loss_history, dtype=np.float64)


    return theta, final_loss, weights_history, loss_history, metrics(X, y, theta)


def confusion_elements(X, y, theta):
    probas = [h_function(x, theta) for x in X]
    predicted_classes = pd.DataFrame({"Predicted_class" : [(1 if x >= 0.5 else 0) for x in probas]})


    predicted_classes = np.array(predicted_classes).flatten()
    y = np.array(y).flatten()

    TP = np.sum((predicted_classes == 1) & (y == 1))
    TN = np.sum((predicted_classes == 0) & (y == 0))
    FP = np.sum((predicted_classes == 1) & (y == 0))
    FN = np.sum((predicted_classes == 0) & (y == 1))

    return TP, TN, FP, FN


def metrics(X, y, theta):

    TP, TN, FP, FN = confusion_elements(X, y, theta)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
    return metrics
