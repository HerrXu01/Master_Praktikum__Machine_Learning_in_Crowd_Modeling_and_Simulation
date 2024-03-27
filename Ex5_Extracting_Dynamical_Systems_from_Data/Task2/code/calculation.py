import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def estimate_matrix(x0, x1, dt):
    """
    Estimate the vectors v and approximate the matrix A.

    Parameters:
    - x0: initial dataset
    - x1: next dataset
    - dt: time step between x0 and x1

    Returns:
    - A: estimated matrix A
    - v: vector v(k)
    """
    v = (x1 - x0) / dt
    A = np.linalg.lstsq(x0, v, rcond=None)[0]
    return A,v


def x1_pre_solve(x0, end_time, A):
    """
    Calculate x1 predicted value

    Parameters:
    - x0: initial dataset
    - end_time: end time for integration
    - A: estimated matrix A

    Returns:
    - x1_pre: prediction of x1 solved by integration
    """
    x1_pre = np.zeros(x0.shape)
    linear_system = lambda t, x, A: A @ x
    for i in range(len(x0)):
        x1_pre[i, :] = solve_ivp(linear_system, t_span=[0, end_time], y0=x0[i, :], t_eval=[end_time],args=[A])["y"].reshape(2,)
    return x1_pre