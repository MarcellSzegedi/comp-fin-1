# import yfinance as yf
# import datetime

# spx_symbol = "^SPX"
# today = "2025-03-28"  # Keep this fixed in your implementation
# end_date = datetime.datetime.strptime(today, "%Y-%m-%d").date()
# start_date = end_date - datetime.timedelta(days=365)

# spx_data = yf.download(spx_symbol, start=start_date, end=end_date)
# lastBusDay = spx_data.index[-1]
# vix_data = yf.download("^VIX", start=lastBusDay, end=lastBusDay + datetime.timedelta(days=1))
# print(vix_data)



# spx_ticker = yf.Ticker("^SPX")
# # Suppose the next expiration is "2025-04-28"
# expiry_date = "2025-04-28"  # Fixed to approximate a 30-day horizon as per CBOE
# chain = spx_ticker.option_chain(expiry_date)
# calls_df = chain.calls
# puts_df = chain.puts

# print("Calls Head:")
# print(calls_df.head())

# print("Puts Head:")
# print(puts_df.head())

# # Optionally, save to CSV
# calls_df.to_csv("spx_calls.csv", index=False)
# puts_df.to_csv("spx_puts.csv", index=False)

import yfinance as yf
import datetime
import numpy as np
import pandas as pd

# Set dates
today_str = "2025-03-28"
end_date = datetime.datetime.strptime(today_str, "%Y-%m-%d").date()
start_date = end_date - datetime.timedelta(days=365)

# Download SPX and VIX data
spx_data = yf.download("^SPX", start=start_date, end=end_date)
lastBusDay = spx_data.index[-1].date()

# Download SPX options chain
ticker = yf.Ticker("^SPX")
expiry_date = "2025-04-28"  # Approx 30-day maturity
calls = ticker.option_chain(expiry_date).calls
puts = ticker.option_chain(expiry_date).puts

# Merge calls and puts into a single dataframe
options = pd.merge(calls, puts, on='strike', how='outer', suffixes=('_call', '_put'))
options = options.dropna(subset=['lastPrice_call', 'lastPrice_put'])  # drop NaNs

# Calculate forward price approximation
# Use put-call parity: F ≈ K + e^(rτ) * (C - P)
tau = 30 / 365
r = 0.05  # risk-free rate (example)
options['mid'] = (options['lastPrice_call'] + options['lastPrice_put']) / 2
atm_strike = options.iloc[(options['mid'] - options['mid'].min()).abs().argsort()[:1]]['strike'].values[0]
F = atm_strike  # Approximate forward price

# Separate OTM options
puts_otm = options[options['strike'] < F]
calls_otm = options[options['strike'] > F]

# Estimate VIX using discretized formula
def vix_integral(df, price_col):
    strikes = df['strike'].values
    prices = df[price_col].values
    delta_K = np.diff(strikes)
    delta_K = np.append(delta_K, delta_K[-1])  # repeat last spacing

    integrand = prices * delta_K / (strikes ** 2)
    return integrand.sum()

put_contrib = vix_integral(puts_otm, 'lastPrice_put')
call_contrib = vix_integral(calls_otm, 'lastPrice_call')

vix_squared = (2 * np.exp(r * tau) / tau) * (put_contrib + call_contrib)
vix_estimate = 100 * np.sqrt(vix_squared)

print(f"Estimated VIX: {vix_estimate:.2f}")

# Get the quoted VIX for comparison
vix_data = yf.download("^VIX", start=lastBusDay, end=lastBusDay + datetime.timedelta(days=1))
vix_quote = vix_data['Adj Close'].iloc[0]
print(f"CBOE VIX quote: {vix_quote:.2f}")


