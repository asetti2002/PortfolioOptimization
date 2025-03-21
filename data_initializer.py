import pandas as pd
import numpy as np
import yfinance as yf
import talib

tickers = ["^GSPC", "^GDAXI", "^IXIC", "^RUT", "^N225", "^FTSE"]
start_date = "2010-01-01"
end_date = "2025-03-20" # change this to whatever current date is

data = {}

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        print(f"No data for {ticker}. Skipping.")
        continue

    # Explicitly convert columns to 1D numpy arrays of type float
    high = np.asarray(df['High'], dtype=float).ravel()
    low = np.asarray(df['Low'], dtype=float).ravel()
    close = np.asarray(df['Close'], dtype=float).ravel()
    volume = np.asarray(df['Volume'], dtype=float).ravel()

    # Optionally, check the shapes
    # print(f"{ticker} shapes -> High: {high.shape}, Low: {low.shape}, Close: {close.shape}")

    # Calculate technical indicators using these arrays
    try:
        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    except Exception as e:
        print(f"Error computing AROONOSC for {ticker}: {e}")
    
    try:
        df['RSI'] = talib.RSI(close, timeperiod=14)
    except Exception as e:
        print(f"Error computing RSI for {ticker}: {e}")

    try:
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    except Exception as e:
        print(f"Error computing CCI for {ticker}: {e}")

    try:
        df['CMO'] = talib.CMO(close, timeperiod=14)
    except Exception as e:
        print(f"Error computing CMO for {ticker}: {e}")

    try:
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    except Exception as e:
        print(f"Error computing MFI for {ticker}: {e}")

    try:
        df['Williams_%R'] = talib.WILLR(high, low, close, timeperiod=14)
    except Exception as e:
        print(f"Error computing Williams %R for {ticker}: {e}")

    try:
        fastk, fastd = talib.STOCHF(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0)
        df['STOCHF_fastk'] = fastk
        df['STOCHF_fastd'] = fastd
    except Exception as e:
        print(f"Error computing STOCHF for {ticker}: {e}")
    
    data[ticker] = df


combined_data = pd.concat(data.values(), axis=1, keys=data.keys())


combined_data.to_pickle('RLdata.pkl')