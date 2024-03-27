import numpy as np
from make_plot import *

class TimeDelayEmbedding():
    def __init__(self, data: np.ndarray):
        """
        Initialize the TimeDelayEmbedding class with a given dataset.
        :param data: A numpy array of data to be used for time delay embedding.
        """
        self.data = data

    def timedelay(self, col: int, n: int, target_dim: int):
        """
        Generate time-delay coordinates from the data.
        :param col: The column of the data array to use.
        :param n: The delay step size.
        :param target_dim: The target dimension (2 or 3) for the output coordinates.
        :return: A numpy array of time-delay coordinates.
        """
        if target_dim not in [2, 3]:
            raise ValueError("The argument target_dim must be 2 or 3.")

        if target_dim == 2:
            delay_coords = np.vstack((self.data[n:, col], self.data[:-n, col])).T
        else:
            c1 = self.data[2*n:, col]
            c2 = self.data[n:-n, col]
            c3 = self.data[:-2*n, col]
            delay_coords = np.column_stack((c1, c2, c3))

        return delay_coords

    def plot_delay(self, delay_coords: np.ndarray, title: str, x_label='X', y_label='Y', z_label='Z'):
        """
        Plot the time-delay coordinates.
        :param delay_coords: A numpy array of time-delay coordinates.
        :param title: Title of the plot.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis.
        :param z_label: Label for the z-axis.
        """
        ddim = delay_coords.shape[1]
        if ddim not in [2, 3]:
            raise ValueError("The number of input columns of the input must be 2 or 3.")
        
        plot_curve(delay_coords, title, x_label=x_label, y_label=y_label, z_label=z_label)
        
