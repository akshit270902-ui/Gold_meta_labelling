import numpy as np
import pandas as pd
from config import BARRIER_MULT


def week_end_index(data: pd.DataFrame, start_idx: int) -> int:
    start_week = data['time'].iloc[start_idx].isocalendar()[1]
    start_year = data['time'].iloc[start_idx].isocalendar()[0]
    end_idx = start_idx
    for i in range(start_idx, len(data)):
        iso = data['time'].iloc[i].isocalendar()
        if iso[0] != start_year or iso[1] != start_week:
            return i - 1
        end_idx = i
    return end_idx


def label_two_barrier(data: pd.DataFrame) -> pd.DataFrame:
    n = len(data)
    labels = np.full(n, np.nan)
    barrier_hit = np.full(n, "", dtype=object)
    bars_to_exit = np.full(n, np.nan)

    for i in range(n):
        signal = data['Signal'].iloc[i]
        if signal == 0:
            continue

        atr = data['ATR'].iloc[i]
        entry_price = data['close'].iloc[i]
        if pd.isna(atr) or pd.isna(entry_price) or atr == 0:
            continue

        vertical_idx = week_end_index(data, i)
        if vertical_idx <= i:
            continue

        if signal == 1:
            upper = entry_price + BARRIER_MULT * atr
            lower = entry_price - BARRIER_MULT * atr
        else:
            upper = entry_price - BARRIER_MULT * atr
            lower = entry_price + BARRIER_MULT * atr

        outcome = np.nan
        exit_offset = np.nan

        for j in range(i + 1, vertical_idx + 1):
            high = data['high'].iloc[j]
            low = data['low'].iloc[j]

            if signal == 1:
                hit_upper = high >= upper
                hit_lower = low <= lower
            else:
                hit_upper = low <= upper
                hit_lower = high >= lower

            if hit_upper and hit_lower:
                outcome = 1
                exit_offset = j - i
                barrier_hit[i] = "both_same_bar_favorable_assumed"
                break
            elif hit_upper:
                outcome = 1
                exit_offset = j - i
                barrier_hit[i] = "favorable"
                break
            elif hit_lower:
                outcome = 0
                exit_offset = j - i
                barrier_hit[i] = "unfavorable"
                break
        else:
            vertical_close = data['close'].iloc[vertical_idx]
            if signal == 1:
                favorable_close = vertical_close > entry_price
            else:
                favorable_close = vertical_close < entry_price
            outcome = 1 if favorable_close else -1
            exit_offset = vertical_idx - i
            barrier_hit[i] = "vertical_favorable" if favorable_close else "vertical_unfavorable"

        labels[i] = outcome
        bars_to_exit[i] = exit_offset

    data['Label'] = labels
    data['BarrierHit'] = barrier_hit
    data['BarsToExit'] = bars_to_exit
    return data
