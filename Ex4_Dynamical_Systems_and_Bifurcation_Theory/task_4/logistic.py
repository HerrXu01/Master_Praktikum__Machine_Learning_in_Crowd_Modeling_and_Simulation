import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def logistic_map(r, x):
    """
    Defines the logistic map equation.
    :param r: Growth rate parameter.
    :param x: Current value.
    :return: Updated value based on the logistic map equation.
    """
    return r * x * (1 - x)

def simulate(r: float, x: np.ndarray, iterations: int = 100, print_x: bool = False, use_plot: bool = True):
    """
    Simulates the logistic map for a given growth rate and initial conditions.
    :param r: Growth rate parameter (0 < r <= 4).
    :param x: Initial conditions (array of values).
    :param iterations: Number of iterations to simulate.
    :param print_x: Flag to print the resulting values.
    :param use_plot: Flag to plot the resulting values.
    :raises ValueError: If r is not in the valid range.
    """
    if r <= 0 or r > 4:
        raise ValueError("The parameter r must satisfy 0<r<=4.")        

    x = x[:, None]
    xn = x
    for i in range(iterations):
        xn = logistic_map(r, xn)
        x = np.concatenate((x, xn), axis=1)
    
    if print_x:
        print(x)
    
    if use_plot:
        for i in range(x.shape[0]):
            plt.plot(np.arange(iterations+1), x[i])
        plt.xlabel("Iterations")
        plt.ylabel("Values")
        plt.title(f"r={r}")
        plt.show()
    
def simulate_animation(initial_values: np.ndarray, iterations: int = 100):
    """
    Creates an animation showing the evolution of the logistic map over time for different r values.
    :param initial_values: Array of initial values to simulate.
    :param iterations: Number of iterations for each simulation.
    """
    r_values = np.linspace(0, 4, 401)
    data = np.zeros((len(r_values), len(initial_values), iterations))

    for r_index, r in enumerate(r_values):
        for x_index, x0 in enumerate(initial_values):
            x = x0
            for n in range(iterations):
                x = logistic_map(r, x)
                data[r_index, x_index, n] = x

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot([], [], lw=2)[0] for _ in initial_values]
    ax.set_xlim(0, iterations - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Logistic Map Iterations')
    r_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        for line in lines:
            line.set_data([], [])
        r_text.set_text('')
        return lines + [r_text]

    def animate(i):
        for j, line in enumerate(lines):
            y_data = data[i, j]
            x_data = np.arange(iterations)
            line.set_data(x_data, y_data)
        r_text.set_text(f'r = {r_values[i]:.2f}')
        return lines + [r_text]

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(r_values), interval=50, blit=True)
    ani.save('logistic_map_animation.gif', writer='pillow', fps=20)
    plt.show()

def logistic_bifurcation(iterations: int = 100):
    """
    Generates and saves a bifurcation diagram of the logistic map.
    :param iterations: Number of iterations for the diagram.
    """
    r_values = np.linspace(0, 4, 401)
    x = np.linspace(0, 0.999, 1000)
    data = np.zeros((iterations+1, len(r_values), len(x)))
    data[0, :, :] = np.tile(x, (len(r_values), 1))

    for it in range(1, iterations+1):
        for r_idx, r in enumerate(r_values):
            xn = logistic_map(r, data[it-1, r_idx, :])
            data[it, r_idx, :] = xn

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Logistic Map Bifurcation Diagram')
    scatter, = ax.plot([], [], 'b.', ms=0.1)

    def animate(i):
        scatter.set_data(np.tile(r_values[:, None], (1, len(x))).reshape(1, -1), data[i, :, :].reshape(1, -1))
        return scatter,

    ani = animation.FuncAnimation(fig, animate, frames=iterations+1, interval=50)
    ani.save('logistic_map_bifurcation_diagram.gif', writer='pillow', fps=20)

    plt.scatter(np.tile(r_values[:, None], (1, len(x))).reshape(1, -1), data[-1, :, :].reshape(1, -1), s=0.1)
    plt.savefig('logistic_map_bifurcation_diagram.png', dpi=300)

    plt.show()

