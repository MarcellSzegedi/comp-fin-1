# Consider a short position in a European call option on a non-dividend paying stock with a maturity of one year
# and strike K = 99 EUR. Let the one-year risk-free interest rate be 6% and the current stock price be 100 EUR.
# Furthermore, assume that the volatility is 20%.
# Use the Euler method to perform a hedging simulation

# Matching Volatility: Conduct an experiment where the volatility in the stock price process matches the
# volatility used in the delta computation (i.e., both set to 20%). Vary the frequency of hedge adjustments
# (from daily to weekly) and explain the results.

import matplotlib.pyplot as plt
import numpy as np


def simulate_euler_black_scholes(S0, mu, sigma, T, N):  # noqa
    """Taken from psuedocode in lecturenotes."""
    dt = T / N  # Time step
    time = np.linspace(0, T, N + 1)  # Time grid
    S = np.zeros(len(time))
    S[0] = S0

    for t in range(1, len(time)):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        S[t] = S[t - 1] + mu * S[t - 1] * dt + sigma * S[t - 1] * dW  # Euler update

    return time, S


# Parameters
S0 = 100  # Initial asset price
mu = 0.05  # Drift
sigma = 0.2  # Volatility
T = 1.0  # Time to maturity
N = 1000  # Number of steps

time, S = simulate_euler_black_scholes(S0, mu, sigma, T, N)

plt.figure(figsize=(12, 6))
plt.plot(time, S)
plt.title("Euler Approximation of BS")
plt.ylabel("Time")
plt.xlabel("Asset price")
plt.savefig("results/euler", dpi=300, bbox_inches="tight")
