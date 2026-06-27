import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH, MODEL_PATH, FEATURE_COLUMNS, TEST_SIZE
from src.utils import load_data
from src.features import create_features, find_signals
from src.labeling import label_two_barrier
from src.backtest import run_backtest, summarize_trades


def main():
    raw = load_data(RAW_DATA_PATH)
    full_data = create_features(raw)
    full_data = find_signals(full_data)

    dataset = label_two_barrier(full_data.copy())
    dataset = dataset[dataset['Signal'] != 0].copy()
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ['Label'])

    X = dataset[FEATURE_COLUMNS]
    _, X_test = train_test_split(X, test_size=TEST_SIZE, shuffle=False)
    test_start_idx = X_test.index.min()

    model = joblib.load(MODEL_PATH)
    print(f"Backtest window starts at row index {test_start_idx} ({full_data['time'].iloc[test_start_idx]})")

    trades_raw = run_backtest(full_data, test_start_idx, model=None)
    trades_filtered = run_backtest(full_data, test_start_idx, model=model)

    summarize_trades(trades_raw, "Run 1: Raw strategy (no LGBM filter)")
    summarize_trades(trades_filtered, "Run 2: LGBM-filtered strategy")


if __name__ == "__main__":
    main()
