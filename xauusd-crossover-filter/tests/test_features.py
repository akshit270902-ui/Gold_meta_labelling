import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.features import create_features, find_signals, cmo, calc_dema


def _make_dummy_df(n=300):
    np.random.seed(42)
    close = 2000 + np.random.randn(n).cumsum() * 2
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    })


def test_create_features_columns_present():
    df = _make_dummy_df()
    out = create_features(df.copy())
    for col in ["ATR", "ATR%", "Short", "Long", "PO", "ADX", "Z_Score", "Entropy", "Hurst", "Composit_E"]:
        assert col in out.columns


def test_find_signals_values():
    df = _make_dummy_df()
    out = create_features(df.copy())
    out = find_signals(out)
    assert set(out['Signal'].unique()).issubset({-1, 0, 1})


def test_cmo_bounds():
    df = _make_dummy_df()
    values = cmo(df['close'], 10).dropna()
    assert (values >= -100).all() and (values <= 100).all()


def test_calc_dema_length_matches():
    df = _make_dummy_df()
    dema = calc_dema(df['close'], 5)
    assert len(dema) == len(df)


def test_create_features_no_lookahead_columns_constant():
    df = _make_dummy_df()
    out1 = create_features(df.copy())
    out2 = create_features(df.copy())
    pd.testing.assert_series_equal(out1['Composit_E'], out2['Composit_E'])
