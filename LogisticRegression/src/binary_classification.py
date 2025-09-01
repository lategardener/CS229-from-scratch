import numpy as np
import pandas as pd


# ==========================================================
# Hypothesis function (sigmoid activation)
# ==========================================================
def h_function(x, theta):
    """
    Compute the sigmoid of the linear combination θ^T * x.
    (i.e., the hypothesis function for logistic regression)

    Parameters:
    - x: feature vector (shape: [n_features])
    - theta: weights vector (shape: [n_features, 1])

    Returns:
    - sigmoid activation (float, between 0 and 1)
    """
    return 1 / (1 + np.exp(-(theta.T @ x)))


# ==========================================================
# Log-likelihood function
# ==========================================================
def log_likelihood(X, y, theta):
    """
    Compute the log-likelihood of the data under the
    logistic regression model.

    Parameters:
    - X: feature matrix (shape: [n_samples, n_features])
    - y: target vector (shape: [n_samples, 1])
    - theta: weights vector (shape: [n_features, 1])

    Returns:
    - log-likelihood value (float) → higher is better
    """
    return np.sum([
        y[i] * np.log(h_function(X[i], theta)) +
        (1 - y[i]) * np.log(1 - h_function(X[i], theta))
        for i in range(len(X))
    ])


def negative_log_likelihood(X, y, theta):
    """
    Compute the negative log-likelihood.
    (This is what we minimize during training)
    """
    return -log_likelihood(X, y, theta)


# ==========================================================
# Prediction error for a single data point
# ==========================================================
def error(X, y, theta, i):
    """
    Compute the prediction error for the i-th data point.

    Parameters:
    - X: feature matrix
    - y: target vector
    - theta: weights vector
    - i: index of the data point

    Returns:
    - error = actual label - predicted probability
    """
    return y[i] - h_function(X[i], theta)


# ==========================================================
# Gradient Descent Algorithm
# Supports: BGD, SGD, MBGD
# ==========================================================
def gradient_descent_ascent(
        X, y,
        epsilon=0.001, max_iter=1000,
        type="BGD", alpha=1e-4, batch_size=32,
        loss_function=negative_log_likelihood
):
    """
    Perform gradient descent to optimize logistic regression weights.

    Parameters:
    - X: feature matrix (shape: [n_samples, n_features], must include intercept column if needed)
    - y: target vector (shape: [n_samples, 1])
    - epsilon: convergence threshold (stop if loss change < epsilon)
    - max_iter: maximum number of iterations
    - type: "BGD" (Batch GD), "SGD" (Stochastic GD), or "MBGD" (Mini-Batch GD)
    - alpha: learning rate
    - batch_size: size of mini-batches (used only if type="MBGD")
    - loss_function: cost function (negative_log_likelihood by default)

    Returns:
    - theta: final weights vector
    - final_loss: last cost value after training
    - weights_history: trajectory of weight vectors during training
    - loss_history: trajectory of loss values during training
    - metrics: evaluation metrics (accuracy, precision, recall, F1-score)
    """
    # Initialize weights to zeros
    theta = np.zeros((X.shape[1], 1))

    # Compute initial cost
    cost = loss_function(X, y, theta)
    weights_history = [theta.copy()]  # track weight updates
    loss_history = [cost]             # track loss evolution

    # Ensure correct loss function
    if loss_function is not negative_log_likelihood and loss_function is not log_likelihood:
        raise ValueError("Loss function must be negative_log_likelihood or log_likelihood")

    # ----------------------
    # Batch GD and Mini-Batch GD
    # ----------------------
    if type in ["BGD", "MBGD"]:
        for _ in range(max_iter):
            for j in range(len(theta)):
                if type == "BGD":
                    # Full dataset gradient
                    grad = np.sum([error(X, y, theta, i) * X[i][j] for i in range(len(X))]) / len(X)
                else:
                    # Mini-batch gradient
                    chosen_data = np.random.choice(len(X), batch_size, replace=False)
                    grad = np.sum([error(X, y, theta, i) * X[i][j] for i in chosen_data]) / batch_size

                # Weight update
                theta[j] = theta[j] + alpha * grad

            # Update cost and track progress
            weights_history.append(theta.copy())
            new_cost = loss_function(X, y, theta)
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
        for i in range(len(X)):
            for j in range(len(theta)):
                grad = error(X, y, theta, i) * X[i][j]
                theta[j] = theta[j] + alpha * grad

            # Track after each update
            weights_history.append(theta.copy())
            cost = loss_function(X, y, theta)
            loss_history.append(cost)

    else:
        raise ValueError("Invalid type. Choose 'BGD', 'SGD', or 'MBGD'.")

    # Final results
    final_loss = loss_history[-1]
    weights_history = np.array([w.flatten() for w in weights_history])
    loss_history = np.array(loss_history, dtype=np.float64)

    return theta, final_loss, weights_history, loss_history, metrics(X, y, theta)


# ==========================================================
# Confusion Matrix Elements
# ==========================================================
def confusion_elements(X, y, theta):
    """
    Compute True Positives, True Negatives, False Positives, and False Negatives.

    Returns:
    - TP, TN, FP, FN
    """
    probas = [h_function(x, theta) for x in X]
    predicted_classes = [(1 if p >= 0.5 else 0) for p in probas]

    predicted_classes = np.array(predicted_classes).flatten()
    y = np.array(y).flatten()

    TP = np.sum((predicted_classes == 1) & (y == 1))
    TN = np.sum((predicted_classes == 0) & (y == 0))
    FP = np.sum((predicted_classes == 1) & (y == 0))
    FN = np.sum((predicted_classes == 0) & (y == 1))

    return TP, TN, FP, FN


# ==========================================================
# Evaluation Metrics
# ==========================================================
def metrics(X, y, theta):
    """
    Compute evaluation metrics for classification:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    """
    TP, TN, FP, FN = confusion_elements(X, y, theta)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
