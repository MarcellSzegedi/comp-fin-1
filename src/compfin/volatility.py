"""
Course: Computational Finance
Names: Marcell ..., Michael ... and Tika van Bennekum
Student IDs: ..., ... and 13392425

File description:
    Part 1 of lab assignment 1.
    In this file we determine the volatility of a certain stock in the markt
    within a certain time. We do this with a classical estimator and a
    Parkinson estimator and compare them. We also compare the results against
    the implied volatility of the same stock.
"""

import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.preprocessing import prices_to_returns


ticker = "TSLA"
start_date = "2015-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")


data = yf.download(ticker, start=start_date, end=end_date)
data.columns = data.columns.droplevel(1)
prices_close = data["Close"]
prices_high = data["High"]
prices_low = data["Low"]


def classical_drift(returns):
    """Formula for natural estimator of the drift parameter mu (Eq 9)."""
    return returns.sum() / len(returns)


def classical_volatility(returns):
    """This is the formula of the classical estimator to calculate volatility (Eq 14).
    In our case, because T=N, it is the same as the std formula from python."""
    returns_squared = returns**2
    mean_return = classical_drift(returns)
    N = len(returns)
    T = N  # timestep is per day, so T = N
    volatility_squared = (1 / (N - 1)) * returns_squared.sum() - (
        T / (N - 1)
    ) * mean_return**2
    volatility = np.sqrt(volatility_squared)
    return volatility


def parkinson_volatility(high_prices, low_prices):
    """Parkinson estimator for volatility."""
    inside_sum = np.log(high_prices / low_prices) ** 2
    constant = 1 / (4 * np.log(2))
    volatility = np.sqrt(constant * inside_sum.sum())
    return volatility


def rolling_window_comparison(returns, prices_high, prices_low):
    """Comparison of a rolling window using a Classical estimator versus a Parkinson estimator."""
    size_window = 30

    # Parkinson rolling window
    inside_sum_parkinson = np.log(prices_high / prices_low) ** 2
    constant_parkinson = 1 / (4 * np.log(2))
    sum_parkinson_window = inside_sum_parkinson.rolling(window=size_window).sum()
    rolling_window_parkinson = np.sqrt(255) * np.sqrt(constant_parkinson * sum_parkinson_window)

    # Classical rolling window
    rolling_window_classical = np.sqrt(255) * returns.rolling(window=size_window).apply(
        classical_volatility, raw=True
    )

    plt.figure(figsize=(12, 6))
    rolling_window_classical.plot(label="Rolling volatility classical")
    rolling_window_parkinson.plot(label="Rolling volatility Parkinson")
    plt.legend()
    plt.title("Rolling volatility: classical vs Parkinson (30 days)")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.savefig("results/classical_vs_parkinson.png", dpi=300, bbox_inches="tight")


def realized_vs_implied():
    """Plots the implied volatility against the classical and Parkinson realized volatility."""
    size_window = 30

    # Load historical price data and implied volatility data
    prices = load_sp500_dataset()
    stock = "GE"  # GE is a random stock from S&P 500

    # Implied volatility
    implied_vol = load_sp500_implied_vol_dataset()["2015":]
    implied_vol = implied_vol[stock]

    # Realized classical volatility
    X = prices_to_returns(prices)
    X = X.loc["2015":][stock]
    realized_vol_clas = X.rolling(window=size_window).apply(
        classical_volatility, raw=True
    )

    # Realized Parkinson volatility
    # extra_data = yf.download(stock, start="2015-01-01", end="2022-12-31")
    # prices_high = extra_data["High"]
    # prices_low = extra_data["Low"]

    # inside_sum_parkinson = np.log(prices_high / prices_low) ** 2
    # constant_parkinson = 1 / (4 * np.log(2))
    # sum_parkinson_window = inside_sum_parkinson.rolling(window=size_window).sum()
    # realized_vol_park = np.sqrt(constant_parkinson * sum_parkinson_window)

    plt.figure(figsize=(12, 6))
    plt.plot(realized_vol_clas * np.sqrt(255), label="Realized Volatility (classical)")
    plt.plot(implied_vol, label="Implied Volatility")
    # plt.plot(realized_vol_park, label="Realized Volatility (Parkinson)")
    plt.legend()
    plt.title(f"Realized vs. implied Volatility for {stock}")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.savefig("results/realized_vs_implied", dpi=300, bbox_inches="tight")


# Alternative (parkinson) volatility
volatility_parkinson = parkinson_volatility(prices_high, prices_low)
# Rolling window comparison
returns = (prices_close - prices_close.shift(1)) / prices_close.shift(1)
rolling_window_comparison(returns, prices_high, prices_low)
# Implied versus realized volatility
realized_vs_implied()

