import numpy as np

# ===========================
# Hypothesis function
# ===========================
def h_function(x, w):
    """
    Compute the predicted value for input x given weights w.
    
    Parameters:
    - x: feature vector (including intercept term)
    - w: weights vector
    
    Returns:
    - prediction (scalar)
    """
    return (w.T @ x).item()

# ===========================
# Error for a single data point
# ===========================
def error(X, y, w, i):
    """
    Compute the prediction error for the i-th data point.
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - w: weights vector
    - i: index of the data point
    
    Returns:
    - prediction error for point i
    """
    return h_function(X[i], w) - y[i]

# ===========================
# Cost function (Mean Squared Error / 2)
# ===========================
def cost_function(X, y, w):
    """
    Compute the cost (mean squared error) over the dataset.
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - w: weights vector
    
    Returns:
    - cost value (scalar)
    """
    return np.sum([error(X, y, w, i) ** 2 for i in range(len(X))]) / (2 * len(X))

# ===========================
# Gradient Descent Algorithm
# Supports: BGD, SGD, MBGD
# ===========================
def gradient_descent(X, y, epsilon=0.001, max_iter=1000, type="BGD", alpha=1e-4, batch_size=32):
    """
    Perform gradient descent to minimize the cost function.
    
    Parameters:
    - X: feature matrix (including intercept term)
    - y: target vector
    - epsilon: convergence threshold
    - max_iter: maximum number of iterations
    - type: "BGD", "SGD", or "MBGD"
    - alpha: learning rate
    - batch_size: size of mini-batches (only for MBGD)
    
    Returns:
    - w: final weights vector
    - final_loss: final cost value
    - weights_history: array of all weight vectors during training
    - loss_history: array of all cost values during training
    """
    # Initialize weights as zeros
    w = np.zeros((X.shape[1], 1))

    # Compute initial cost
    cost = cost_function(X, y, w)
    weights_history = [w.copy()]  # store initial weights
    loss_history = [cost]         # store initial cost

    # ----------------------
    # Batch or Mini-Batch Gradient Descent
    # ----------------------
    if type == "BGD" or type == "MBGD":
        for _ in range(max_iter):
            for j in range(len(w)):
                if type == "BGD":
                    # Compute gradient over the whole dataset
                    grad = np.sum([error(X, y, w, i) * X[i][j] for i in range(len(X))]) / len(X)
                else:
                    # Mini-batch gradient
                    chosen_data = np.random.choice(len(X), batch_size, replace=False)
                    grad = np.sum([error(X, y, w, i) * X[i][j] for i in chosen_data]) / batch_size
                # Update weight
                w[j] = w[j] - alpha * grad

            # Store updated weights and cost
            weights_history.append(w.copy())
            new_cost = cost_function(X, y, w)
            loss_history.append(new_cost)

            # Check convergence
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
            for j in range(len(w)):
                grad = error(X, y, w, i) * X[i][j]
                w[j] = w[j] - alpha * grad
            weights_history.append(w.copy())
            cost = cost_function(X, y, w)
            loss_history.append(cost)
    else:
        raise ValueError("Invalid gradient descent type")

    # Final results
    final_loss = loss_history[-1]
    weights_history = np.array([w.flatten() for w in weights_history])
    loss_history = np.array(loss_history, dtype=np.float64)

    theta0, theta1 = w[0].item(), w[1].item()
    print(f"Final weights : \nθ₀ = {theta0}\nθ₁ = {theta1}")
    print(f"Final loss : {final_loss}")

    return w, final_loss, weights_history, loss_history
