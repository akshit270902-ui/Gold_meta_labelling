# XAUUSD Crossover Strategy with LGBM Trade Filter

A two-stage system for trading XAUUSD on 1-hour bars:

- A **rule-based crossover strategy** (SMA 3 / SMA 9) opens long or short positions, with both barriers sized at 3x the entry-bar ATR.
- A **two-barrier (vertical-capped) labeling scheme** assigns each historical crossover a favorable (1), unfavorable (0), or vertical-timeout outcome (favorable close = 1, unfavorable close = -1), where the vertical barrier is forced at the end of the trading week.
- An **LGBM classifier** is trained on engineered technical features to predict whether a crossover signal is likely to resolve favorably, acting as a trade filter on top of the rule-based signal.

---

## Method

### Signal Generation

A crossover between the 3-period and 9-period SMA of `close` triggers a directional signal:

- Cross up → long
- Cross down → short

### Two-Barrier Labeling

For each signal bar, the upper and lower barriers are placed at `entry_price ± 3 x ATR(8)`, direction-aware. Bars are scanned forward until a barrier is hit, or until the last bar of the trading week (vertical barrier) is reached.

- Upper/favorable barrier hit first → label `1`
- Lower/unfavorable barrier hit first → label `0`
- Vertical barrier reached with no price barrier hit → label `1` if the close is favorable relative to entry, else `-1`

### Model

`LGBMClassifier` trained with a multiclass objective over the three label classes, on a chronological 70/30 train/test split (no shuffling, to avoid look-ahead leakage).

### Backtest

Two single-position backtests are run over the held-out test period:

- **Run 1** — every crossover opens a trade, but a new signal is ignored while a position is open.
- **Run 2** — identical, except a crossover only opens a trade if the trained model predicts a favorable outcome.

---

## Quickstart

```bash
pip install -r requirements.txt
```

Place your OHLCV CSV at `data/xauusd.csv` (must contain `time, open, high, low, close, volume`). Then:

```bash
python scripts/train.py
python scripts/backtest.py
```

---

## Project Structure

```
src/features.py    feature engineering and crossover signal detection
src/labeling.py    two-barrier labeling with end-of-week vertical cutoff
src/model.py       LGBM training and single-row inference
src/backtest.py    single-position backtest engine, raw and filtered
src/utils.py       data loading
scripts/train.py   builds dataset, trains model, saves artifact
scripts/backtest.py runs both backtests against the saved model
tests/             unit tests for feature and signal logic
```

---

## Notes

- Barrier distance is `3 x ATR(8)` in raw price units, matching the original live strategy's risk parameters.
- The vertical barrier always falls at the last bar before the weekly market close, so trades are never held over the weekend gap.
- The LGBM filter rejects a signal unless the predicted class is strictly favorable (`1`); a predicted favorable-timeout (`-1`) is not treated as sufficient to enter.

## License

MIT
