import pandas as pd
import numpy as np
import torch


def get_price_window(df: pd.DataFrame, ref_date: str, lookback_window: int) -> pd.DataFrame:
    _ = df[df.index <= ref_date].sort_index()
    return _ if lookback_window < 0 else _.tail(lookback_window)


def decompose_weights_df(df_weights: pd.DataFrame) -> tuple:
    deltas = df_weights.diff()
    w_b = np.abs(deltas * (deltas >= 0))
    w_s = deltas * (deltas <= 0)
    w_h = df_weights * (df_weights <= df_weights.shift(1)) + df_weights.shift(1) * (df_weights > df_weights.shift(1))
    return w_h, w_b, w_s


def decompose_weights_tensor(w_n: torch.Tensor, w_o: torch.Tensor) -> tuple:
    w_delta = w_n - w_o
    w_b = torch.abs(w_delta * (w_delta > 0)).float()
    w_s = torch.abs(w_delta * (w_delta < 0)).float()
    w_h = 0.5*(w_o+w_n-w_s-w_b).float()
    assert torch.max(torch.abs(w_h+w_b-w_n))<1e-6
    return w_h, w_b, w_s


def compute_OHLC_returns(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    df_rets = df_ohlc.copy(deep = True)
    df_rets['r_b'] = (df_rets['Close'] / df_rets['Open']) - 1
    df_rets['r_s'] = (df_rets['Open'] / df_rets['Close'].shift(1)) - 1
    df_rets['r_h'] = (df_rets['Close'] / df_rets['Close'].shift(1)) - 1

    return df_rets.loc[:, ['r_h', 'r_b', 'r_s']]


def create_return_matrices(df_instruments: pd.DataFrame) -> tuple:
    r_h = dict()
    r_b = dict()
    r_s = dict()
    for i in set([i[0] for i in df_instruments.columns]):
        _ = compute_OHLC_returns(df_instruments[i])
        r_h[i] = _['r_h'].values
        r_b[i] = _['r_b'].values
        r_s[i] = _['r_s'].values

    r_h = pd.DataFrame(r_h, index=df_instruments.index).dropna()
    r_b = pd.DataFrame(r_b, index=df_instruments.index).dropna()
    r_s = pd.DataFrame(r_s, index=df_instruments.index).dropna()

    return r_h, r_b, r_s


def normalize_price_window(df_price_window: pd.DataFrame, norm_pivot: str = 'window_close') -> pd.DataFrame:
    if norm_pivot == 'prev_day_close':
        df_norm_price = df_price_window.div(df_price_window['Close'].shift(1), axis=0)
    elif norm_pivot == 'day_close':
        df_norm_price = df_price_window.div(df_price_window['Close'], axis=0)
    elif norm_pivot == 'day_open':
        df_norm_price = df_price_window.div(df_price_window['Open'], axis=0)
    elif norm_pivot == 'window_close':
        df_norm_price = df_price_window / df_price_window['Close'][-1]
    elif norm_pivot == 'window_open':
        df_norm_price = df_price_window / df_price_window['Open'][0]
    else:
        raise ValueError(f"{norm_pivot} is not a valid normalization strategy")

    return df_norm_price



def get_weights_asTensors(new_weights, current_weights) -> tuple:
    w_delta = new_weights - current_weights
    buy_weights = torch.clamp(w_delta, min=0.0)
    sell_weights = torch.clamp(-w_delta, min=0.0)
    hold_weights = torch.min(new_weights, current_weights)
    return hold_weights, buy_weights, sell_weights