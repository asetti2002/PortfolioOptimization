import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppose your OHLC data is stored in a multi-indexed DataFrame where
# columns are of the form (Ticker, 'Close'). For example:
#    ^RUT      ^GDAXI   ...    ^GSPC    CASH
#    Open High Low Close Volume   Open High Low Close Volume   ... etc.
#
# We'll assume your DataFrame is already loaded as df_ohlc.
def ewp():
# Convert the index to datetime if it isn't already, and filter to data from Jan 1, 2000 onwards.
    df = pd.read_hdf("data/data.h5", key="instruments")
    df.index = pd.to_datetime(df.index)
    df = df.loc["1990-01-01":].dropna()

    # Get a list of instruments using the 'Close' price columns. 
    # (Adjust as needed if some columns should be excluded, like "CASH")
    instruments = sorted({col[0] for col in df.columns if col[1] == 'Close'})

    # Number of instruments
    n = len(instruments)

    # Compute daily returns for each instrument.
    # We use the formula: return = (Close_t / Close_{t-1}) - 1 
    returns_dict = {}
    for inst in instruments:
        close_prices = df[inst]['Close']
        daily_returns = close_prices.pct_change()  # This computes (Close_t - Close_{t-1}) / Close_{t-1}
        returns_dict[inst] = daily_returns

    # Create a DataFrame of returns (each column is an instrument)
    returns_df = pd.DataFrame(returns_dict)

    # For a long-only equally weighted portfolio, each instrument gets a weight of 1/n.
    weights = np.ones(n) / n

    # It is typically more robust to work with log returns because they aggregate additively.
    # Compute log returns: log(1 + return)
    log_returns_df = np.log(1 + returns_df)

    # Compute the daily portfolio log return as the weighted sum of the instruments' log returns.
    # Note that log returns are additive: log(P_t/P_0) = sum(log returns over time)
    portfolio_log_return = log_returns_df.dot(weights)

    # Clean up NaN values (the first day will be NaN because of the pct_change)
    portfolio_log_return = portfolio_log_return.fillna(0)

    # Compute the cumulative portfolio value:
    # Starting with an initial portfolio value of 1, the cumulative value is:
    # portfolio_value_t = exp(sum(log returns up to time t))
    cumulative_value = np.exp(portfolio_log_return.cumsum())
    return cumulative_value

# Plot the cumulative portfolio value over time.
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_value.index, cumulative_value.values, label='Equally Weighted Portfolio')
# plt.xlabel("Date")
# plt.ylabel("Portfolio Value (Relative)")
# plt.title("Equally Weighted Portfolio Value Over Time")
# plt.legend()
# plt.grid(True)
# plt.show()
