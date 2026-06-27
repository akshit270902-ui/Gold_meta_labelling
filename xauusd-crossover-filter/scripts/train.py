import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
from config import RAW_DATA_PATH, MODEL_PATH, FEATURE_COLUMNS
from src.utils import load_data
from src.features import create_features, find_signals
from src.labeling import label_two_barrier
from src.model import train_model


def main():
    raw = load_data(RAW_DATA_PATH)
    full_data = create_features(raw)
    full_data = find_signals(full_data)

    dataset = label_two_barrier(full_data.copy())
    dataset = dataset[dataset['Signal'] != 0].copy()
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ['Label'])

    print("Total signals labeled:", len(dataset))
    print(dataset['Label'].value_counts())

    model, train_idx, test_idx = train_model(dataset)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("test_start_idx:", test_idx.min())


if __name__ == "__main__":
    main()
