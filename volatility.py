"""
Course: Computational Finance
Names: Marcell ..., Michael ... and Tika van Bennekum
Student IDs: ..., ... and 13392425

File description:
    Part 1 of lab assignment 1.
    ...

Notes:
    tickers:
    "AAPL" is for Apple Inc.
    "MSFT" is for Microsoft
    "GOOGL" is for Alphabet (Google)
    "^GSPC" is for the S&P 500 index
    "TSLA" is for Tesla
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

# Equation 9 and 14 from app A
# since the timesteps are 1, we can use python mean and std formula
returns = (prices_close - prices_close.shift(1)) / prices_close.shift(1)
mean_return = returns.mean()
volatility = returns.std()

# alternative (parkinson) volatility
inside_sum_parkinson = np.log(prices_high / prices_low)**2
constant_parkinson = 1 / 4 * np.log(2)
volatility_parkinson = np.sqrt(constant_parkinson * inside_sum_parkinson.sum())

def rolling_window_comparison():
    size_window = 30
    rolling_window = returns.rolling(window=size_window).std()
    sum_parkinson_window = inside_sum_parkinson.rolling(window=size_window).sum()
    rolling_window_parkinson = np.sqrt(constant_parkinson * sum_parkinson_window)


    plt.figure(figsize=(12, 6))
    rolling_window.plot(label="Rolling volatility classical")
    rolling_window_parkinson.plot(label="Rolling volatility Parkinson")
    plt.legend()
    plt.title("Rolling volatility: classical vs Parkinson (30 days)")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.show()

# rolling_window_comparison()

# volatility signature plot
# Unsure what m stand for, wheter it means the size of the rolling window or something else
# TA also doesn't know, so he will tell me tommorow.
def signature_plot():
    return

def realized_vs_implied():
    # Load historical price data and implied volatility data
    # TODO: results look weird, fix it
    # TODO: add the parkinson estimator
    prices = load_sp500_dataset()
    stock = "GE" # GE is a random stock from S&P 500

    implied_vol = load_sp500_implied_vol_dataset()["2015":]
    implied_vol = implied_vol[stock]

    X = prices_to_returns(prices)
    X = X.loc["2015":][stock]
    realized_vol = X.rolling(window=30).std()

    plt.figure(figsize=(12, 6))
    plt.plot(realized_vol, label="Realized Volatility")
    plt.plot(implied_vol, label="Implied Volatility")
    plt.legend()
    plt.title(f"Realized vs. Implied Volatility for {stock}")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.show()

realized_vs_implied()

