WINDOW_FAST_SMA = 3
WINDOW_SLOW_SMA = 9
ATR_WINDOW = 8
BARRIER_MULT = 3.0

FEATURE_COLUMNS = ['Composit_E', 'PO', 'Week', 'Hour', 'ATR', 'ADX', 'Entropy', 'Z_Score', 'Hurst']
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

TEST_SIZE = 0.30
RANDOM_STATE = 42

LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    random_state=RANDOM_STATE
)

RAW_DATA_PATH = "data/xauusd.csv"
RESULTS_DIR = "results/"
MODEL_PATH = "results/lightgbm.joblib"
