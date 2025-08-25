import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from LinearRegression.src.regression import *


def plot_data(X, y):
    matplotlib.use("module://matplotlib_inline.backend_inline")

    colors = np.random.uniform(15, 80, len(X))
    fig, ax = plt.subplots()

    ax.scatter(X, y, c=colors, marker='.')

    # Add labels and title
    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.set_title("Scatter plot of training data")

    plt.show()



def plot_h_function(X, y, X_intercept, w):
    """
    # Plot the hypothesis function using the final weights over the training data
    """

    matplotlib.use("module://matplotlib_inline.backend_inline")
    image = np.array([h_function(x, w) for x in X_intercept])
    _, ax = plt.subplots()
    ax.scatter(X, y, marker='.')
    ax.plot(image, image, color="red")

    # Add labels and title
    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.set_title("Scatter plot of training data and hypothesis function")

    plt.show()


def plot_weights_trajectory(weights_history):
    """
    # Plot weights_history evolution over time
    """

    _, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight value")

    epochs = len(weights_history)
    x = np.arange(epochs)  # epoch indices

    # Plot weight curves directly
    ax.plot(x, weights_history[:,0], "r.-", label="θ₀")
    ax.plot(x, weights_history[:,1], "b.-", label="θ₁")
    ax.legend()

    # Optionally set axis limits
    ax.set_xlim(0, epochs-1)
    ax.set_ylim(np.min(weights_history)-0.1, np.max(weights_history)+0.1)

    plt.show()

def plot_loss_trajectory(weights_history, loss_history,X_intercept, y):
    """
    # 3D plot of loss landscape and final weight trajectory
    """

    epochs = len(weights_history)

    # --- 3D figure setup ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("w1")  # swapped axis
    ax.set_ylabel("w0")  # swapped axis

    # --- Set axis limits ---
    ax.set_xlim(np.min(weights_history[:,1])-0.1, np.max(weights_history[:,1])+0.1)
    ax.set_ylim(np.min(weights_history[:,0])-0.1, np.max(weights_history[:,0])+0.1)
    ax.set_zlim(np.min(loss_history)-1, np.max(loss_history)+1)

    # --- Create a surface (loss "landscape") ---
    w0_grid = np.linspace(np.min(weights_history[:,0])-0.1, np.max(weights_history[:,0])+0.1, 50)
    w1_grid = np.linspace(np.min(weights_history[:,1])-0.1, np.max(weights_history[:,1])+0.1, 50)
    W1, W0 = np.meshgrid(w1_grid, w0_grid)  # swap axes to match plot

    Z = np.array([[np.sum((X_intercept @ np.array([[w0],[w1]]) - y)**2)/ (2 * len(y))
                   for w0, w1 in zip(row_w0, row_w1)]
                  for row_w0, row_w1 in zip(W0, W1)])

    ax.plot_surface(W1, W0, Z, alpha=0.3, cmap='viridis')

    # --- Loss trajectory
    ax.scatter(weights_history[:,1], weights_history[:,0], loss_history,
               color='black', marker='.', s=20, label='Weight points')

    # Trajectory plot
    ax.plot(weights_history[:,1], weights_history[:,0], loss_history,
            color='red', lw=2, label='Loss trajectory')
    ax.legend()

    plt.show()
