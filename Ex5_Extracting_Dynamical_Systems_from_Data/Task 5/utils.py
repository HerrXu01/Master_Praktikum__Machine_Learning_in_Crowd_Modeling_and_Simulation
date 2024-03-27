
import numpy as np
from scipy.spatial.distance import cdist

def compute_arc_length(x, y, z):
    """
    Compute the total arc length and velocity of a curve defined by discrete points (x, y, z).
    
    Parameters:
        x, y, z: NumPy arrays representing the coordinates of the points.

    Returns:
        Total velocity, arc length of the curve.
    """
    # Calculate the distance between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    dt = 1

    # Compute the arc length, velocity between consecutive points using the Euclidean distance formula
    arclength = np.sqrt(dx**2 + dy**2 + dz**2)
    velocity = np.sqrt(dx**2 + dy**2 + dz**2)/dt

    return arclength, velocity

def rbf(x, xl, eps):
    """
    Computes radial basis function

    Parameters:
        x : points
        xl : center points
        eps : bandwidth of basis function

    Returns:
        kernel matrix containing radial basis functions
    """

    return np.exp(-cdist(x, xl)**2/(eps**2))

def nonlin_func_approx(points, targets, centers, eps):
    """
    Approximates the non-linear func on the data through least squares method

    Parameters:
        points : points on which basis function will be calculated
        targets : target points
        centers : points to compute the basis function
        eps : bandwidth of basis function
    
    Returns:
        kernel matrix, least square soln(coefficient matrix), residual matrix
    
    """
    phi = rbf(points, centers, eps)
    sol, res, rank, singvals = np.linalg.lstsq(phi, targets, rcond=1e-5)
    return phi, sol, res

def rbf_approx(t, y, centers, eps, C):
    """
    function to return the vector field for a single point (rbf)

    Parameters:
        t: time (for solve_ivp)
        y: single point
        centers: all centers
        eps: radius of gaussians
        C: coefficient matrix, found with least squares
    
    Returns: 
        derivative for point y
    """    
    y = y.reshape(1, y.shape[-1])
    phi = np.exp(-cdist(y, centers) ** 2 / eps ** 2)
    return phi @ C






