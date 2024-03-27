import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(data: np.ndarray, title: str, x_label='X Axis', y_label='Y Axis', z_label='Z Axis'):
    """
    Plot a scatter plot of the given data.
    :param data: A numpy array of the data points.
    :param title: Title of the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param z_label: Label for the z-axis (only if 3D).
    """
    if len(data.shape) != 2 or data.shape[1] not in [2, 3]:
        raise ValueError("The input data must be an Numpy array of the shape [n, 2] or [n, 3].")
    
    num_dimensions = data.shape[1]
    if num_dimensions == 2:
        plt.scatter(data[:, 0], data[:, 1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.title(title)

    plt.show()

def plot_curve(data: np.ndarray, title: str, x_label='X', y_label='Y', z_label='Z'):
    """
    Plot a line curve of the given data.
    :param data: A numpy array of the data points.
    :param title: Title of the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param z_label: Label for the z-axis (only if 3D).
    """
    if len(data.shape) != 2 or data.shape[1] not in [2, 3]:
        raise ValueError("The input data must be an Numpy array of the shape [n, 2] or [n, 3].")

    fig = plt.figure(figsize=(10, 8))

    if data.shape[1] == 2:
        ax = fig.add_subplot()
        ax.plot(data[:, 0], data[:, 1])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.show()
    else:
        ax = fig.add_subplot(projection='3d')
        ax.plot(data[:, 0], data[:, 1], data[:, 2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.set_title(title)
        plt.show()

def plot_lorenz_delay(traj: np.ndarray, title: str, x_label: str, y_label: str, z_label: str):
    """
    Plot a trajectory in a 3D space (useful for visualizing dynamics like the Lorenz attractor).
    :param traj: A numpy array of the trajectory points.
    :param title: Title of the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param z_label: Label for the z-axis.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    plt.show()