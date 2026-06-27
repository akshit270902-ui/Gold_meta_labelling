import numpy as np
import pandas as pd
from scipy.stats import entropy
from hurst import compute_Hc
import ta
from config import WINDOW_FAST_SMA, WINDOW_SLOW_SMA, ATR_WINDOW


def rolling_hurst(series: pd.Series, window: int) -> np.ndarray:
    hurst_values = []
    for i in range(len(series)):
        if i < window:
            hurst_values.append(np.nan)
        else:
            window_data = series[i - window:i].dropna()
            if len(window_data) < 2 or np.ptp(window_data) == 0:
                hurst_values.append(np.nan)
            else:
                try:
                    h, _, _ = compute_Hc(window_data, simplified=True)
                    hurst_values.append(h)
                except Exception:
                    hurst_values.append(np.nan)
    return np.array(hurst_values)


def calc_dema(src: pd.Series, length: int) -> pd.Series:
    e1 = src.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    return 2 * e1 - e2


def cmo(src: pd.Series, period: int) -> pd.Series:
    cmo_up = (src.diff().apply(lambda x: max(x, 0))).rolling(window=period).sum()
    cmo_down = (src.diff().apply(lambda x: abs(min(x, 0)))).rolling(window=period).sum()
    return 100 * ((cmo_up - cmo_down) / (cmo_up + cmo_down)).fillna(0)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data['Week'] = data['time'].apply(lambda d: (d.day - 1) // 7 + 1)
    data['Hour'] = data['time'].dt.hour
    data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=ATR_WINDOW)
    data['ATR%'] = data['ATR'] / data['close']
    data['Short'] = ta.trend.sma_indicator(data['close'], window=WINDOW_FAST_SMA)
    data['Long'] = ta.trend.sma_indicator(data['close'], window=WINDOW_SLOW_SMA)
    data['PO'] = (data['Short'] - data['Long']) / data['Long'] * 100
    data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'], window=100)
    data['Z_Score'] = (data['close'] - data['close'].rolling(window=21).mean()) / data['close'].rolling(window=21).std()
    data['Returns'] = data['close'].pct_change()
    data['Entropy'] = data['Returns'].rolling(window=100).apply(lambda x: entropy(np.histogram(x.dropna(), bins=5, density=True)[0]))
    data['Hurst'] = rolling_hurst(data['close'], window=100)
    cmo5 = calc_dema(cmo(data['close'], 5), 3)
    cmo10 = calc_dema(cmo(data['close'], 10), 3)
    cmo20 = calc_dema(cmo(data['close'], 20), 3)
    stdev5 = data['close'].rolling(window=5).std()
    stdev10 = data['close'].rolling(window=10).std()
    stdev20 = data['close'].rolling(window=20).std()
    dmi = ((stdev5 * cmo5) + (stdev10 * cmo10) + (stdev20 * cmo20)) / (stdev5 + stdev10 + stdev20)
    data['Composit_E'] = dmi.ewm(span=3, adjust=False).mean()
    return data


def find_signals(data: pd.DataFrame) -> pd.DataFrame:
    short = data['Short']
    long = data['Long']
    diff = short - long
    prev_diff = diff.shift(1)
    cross_up = (prev_diff <= 0) & (diff > 0)
    cross_down = (prev_diff >= 0) & (diff < 0)
    data['Signal'] = 0
    data.loc[cross_up, 'Signal'] = 1
    data.loc[cross_down, 'Signal'] = -1
    return data
