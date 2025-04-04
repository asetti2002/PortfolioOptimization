import pandas as pd
import numpy as np
import yfinance as yf
import talib
from datetime import datetime



tickers = ["^GSPC", "^GDAXI", "^IXIC", "^RUT", "^N225", "^FTSE"]
start_date = "2000-01-01"
end_date = datetime.today().date()

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




ohlc_labels = ['Close', 'High', 'Low', 'Open', 'Volume']

# Assume 'df' is your unified DataFrame

# For a DataFrame with multi-indexed columns, the level that contains "Price", "Close", etc., is typically level 1.
df_ohlc = combined_data.loc[:, combined_data.columns.get_level_values(1).isin(ohlc_labels)]
df_indicators = combined_data.loc[:, ~combined_data.columns.get_level_values(1).isin(ohlc_labels)]


ohlc_keywords = ['Close', 'High', 'Low', 'Open', 'Volume']
ohlc_cols = [col for col in combined_data.columns if any(keyword in col for keyword in ohlc_keywords)]
indicator_cols = [col for col in combined_data.columns if col not in ohlc_cols]

df_ohlc = combined_data[ohlc_cols]
df_indicators = combined_data[indicator_cols]


combined_data.to_hdf('combined_data.h5', key='combined_data', mode='w')
df_ohlc.to_hdf('ohlc_data.h5', key='ohlc_data', mode='w')
df_indicators.to_hdf('indicators_data.h5', key='indicators_data', mode='w')


# combined_data.to_pickle('RLdata.pkl')