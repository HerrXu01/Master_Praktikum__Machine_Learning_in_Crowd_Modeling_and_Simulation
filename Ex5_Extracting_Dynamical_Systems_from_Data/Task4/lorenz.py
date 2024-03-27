import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorenz_simulate(x0, sigma=10.0, beta=8.0/3.0, rho=28, t=np.linspace(0, 1000, 10000), return_traj=False, use_plot=True):
    """
    Simulate the Lorenz attractor system.

    Parameters:
    x0 : array_like
        Initial state (position in X, Y, Z space).
    sigma : float, optional
        Parameter sigma of the Lorenz system.
    beta : float, optional
        Parameter beta of the Lorenz system.
    rho : float, optional
        Parameter rho of the Lorenz system.
    t : array_like, optional
        Time points at which the solution should be reported.
    return_traj : bool, optional
        Whether to return the trajectory.
    use_plot : bool, optional
        Whether to plot the trajectory.

    Returns:
    array_like (if return_traj is True)
        The trajectory of the system.
    """
    def lorenz_system(state, t):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    trajectory = odeint(lorenz_system, x0, t)
    if return_traj:
        return trajectory

    if use_plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(f"Lorenz Attractor Trajectory with initial point x0 = {x0}.")
        plt.show()


def lorenz_compare(x0, x0_hat, sigma=10.0, beta=8.0/3.0, rho=28, t=np.linspace(0, 1000, 10000)):
    """
    Compare two Lorenz attractor trajectories starting from different initial conditions.

    Parameters:
    x0 : array_like
        Initial state of the first trajectory (position in X, Y, Z space).
    x0_hat : array_like
        Initial state of the second trajectory (position in X, Y, Z space).
    sigma : float, optional
        Parameter sigma of the Lorenz system.
    beta : float, optional
        Parameter beta of the Lorenz system.
    rho : float, optional
        Parameter rho of the Lorenz system.
    t : array_like, optional
        Time points at which the solution should be reported.

    Returns:
    None
        The function plots the comparison of the two trajectories and their difference over time.
    """
    xt = lorenz_simulate(x0=x0, sigma=sigma, beta=beta, rho=rho, t=t, return_traj=True, use_plot=False)
    xt_hat = lorenz_simulate(x0=x0_hat, sigma=sigma, beta=beta, rho=rho, t=t, return_traj=True, use_plot=False)
    difference = np.linalg.norm(xt - xt_hat, axis=1)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], lw=0.5, label='Original')
    ax.plot(xt_hat[:, 0], xt_hat[:, 1], xt_hat[:, 2], lw=0.5, label='Perturbed')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Comparison between the original and perturbed Lorenz Attractor Trajectories")
    ax.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, difference, lw=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("L2 Norm of Difference")
    ax2.set_title("Difference between Trajectories over Time")
    ax2.axhline(y=1, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()

    time_larger_than_1 = t[difference > 1][0] if any(difference > 1) else None
    if time_larger_than_1 is not None:
        print(f"The first time point where the difference of two trajectories is larger than 1 is {time_larger_than_1:.1f}.")
    else:
        print("The difference of all time points is smaller than 1.")
