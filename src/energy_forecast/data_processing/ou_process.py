# OU process simulation

import numpy as np
import matplotlib.pyplot as plt

def ou_process(theta=0.7, mu=0.0, sigma=0.3, X0=1.0, T=30.0, dt=0.01):
    """
    Simulates an Ornstein-Uhlenbeck (OU) process, a stochastic process often
    used to model mean-reverting behavior such as the velocity of a particle
    under friction or financial interest rates.

    The equation governing the OU process is given by:
    dX_t = θ(μ - X_t)dt + σdW_t,
    where dW_t represents a Wiener process.

    :param theta: The speed of mean reversion. Higher values cause
        the process to revert to its mean at a faster rate.
    :param mu: The long-term mean that the process reverts to
        over time.
    :param sigma: The volatility or magnitude of random fluctuations
        in the process.
    :param X0: The initial value of the process.
    :param T: The total simulation time period.
    :param dt: The incremental time step for the simulation.
    :return: A NumPy array representing the OU process values
        over time from t=0 to t=T.
    :rtype: numpy.ndarray
    """
    N = int(T / dt)  # Number of time steps

    # Pre-allocate array for efficiency
    X = np.zeros(N)
    X[0] = X0

    # Generate the OU process
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) * dt + sigma * dW
    return X

def plot_ou_process(X, T, N):
    # Plot the result
    plt.plot(np.linspace(0, T, N), X)
    plt.title("Ornstein-Uhlenbeck Process Simulation")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.show()
