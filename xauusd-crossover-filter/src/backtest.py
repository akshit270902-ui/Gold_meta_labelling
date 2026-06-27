import pandas as pd
from config import BARRIER_MULT
from src.labeling import week_end_index
from src.model import predict_favorable


def run_backtest(data: pd.DataFrame, start_idx: int, model=None) -> pd.DataFrame:
    n = len(data)
    trades = []
    position = None

    for i in range(start_idx, n):
        if position is not None:
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            direction = position['direction']
            upper = position['upper']
            lower = position['lower']

            if direction == 1:
                hit_upper = high >= upper
                hit_lower = low <= lower
            else:
                hit_upper = low <= upper
                hit_lower = high >= lower

            closed = False
            if hit_upper and hit_lower:
                exit_price = upper
                reason = "tp_sl_same_bar"
                closed = True
            elif hit_upper:
                exit_price = upper
                reason = "tp"
                closed = True
            elif hit_lower:
                exit_price = lower
                reason = "sl"
                closed = True
            elif i >= position['vertical_idx']:
                exit_price = data['close'].iloc[i]
                reason = "timeout"
                closed = True

            if closed:
                pnl = (exit_price - position['entry_price']) * direction
                trades.append({
                    "entry_idx": position['entry_idx'],
                    "exit_idx": i,
                    "entry_time": position['entry_time'],
                    "exit_time": data['time'].iloc[i],
                    "direction": direction,
                    "entry_price": position['entry_price'],
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl": pnl,
                })
                position = None

        if position is None:
            signal = data['Signal'].iloc[i]
            if signal != 0:
                atr = data['ATR'].iloc[i]
                entry_price = data['close'].iloc[i]
                vertical_idx = week_end_index(data, i)

                if not pd.isna(atr) and atr != 0 and vertical_idx > i:
                    take_trade = True
                    if model is not None:
                        take_trade = predict_favorable(model, data.iloc[i])

                    if take_trade:
                        if signal == 1:
                            upper = entry_price + BARRIER_MULT * atr
                            lower = entry_price - BARRIER_MULT * atr
                        else:
                            upper = entry_price - BARRIER_MULT * atr
                            lower = entry_price + BARRIER_MULT * atr

                        position = {
                            "entry_idx": i,
                            "entry_time": data['time'].iloc[i],
                            "direction": signal,
                            "entry_price": entry_price,
                            "upper": upper,
                            "lower": lower,
                            "vertical_idx": vertical_idx,
                        }

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame, label: str) -> None:
    print(f"--- {label} ---")
    if len(trades) == 0:
        print("No trades taken.")
        return
    total = len(trades)
    wins = (trades['pnl'] > 0).sum()
    losses = (trades['pnl'] <= 0).sum()
    win_rate = wins / total * 100
    total_pnl = trades['pnl'].sum()
    avg_pnl = trades['pnl'].mean()
    print(f"Total trades: {total}")
    print(f"Wins: {wins}  Losses: {losses}  Win rate: {win_rate:.2f}%")
    print(f"Total PnL (price units): {total_pnl:.4f}")
    print(f"Average PnL per trade: {avg_pnl:.4f}")
    print(trades['reason'].value_counts())
    print()
