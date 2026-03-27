"""
indicators.py — compute all v6 features for the RSI bot

add_indicators(df)      : compute all per-symbol features from OHLCV bars
add_spy_features(df)    : compute SPY-level features from SPY bars
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator


def add_indicators(df: pd.DataFrame, rsi_period: int = 2) -> pd.DataFrame:
    """
    Compute all v6 per-symbol features.

    Input df columns: date, symbol, open, high, low, close, volume
    Adds:
      MAs          : ma_5, ma_20, ma_50, ma_200
      RSI          : rsi_2
      Volume       : vol_ma20, vol_ratio
      Returns      : ret_5d, ret_10d, ret_20d
      ATR          : atr_14, atr_pct
      Pct from MAs : pct_from_ma200, pct_from_ma50, pct_from_ma20
      Intraday     : close_in_range
      Structure    : low_252d, dist_52wk_low
      Momentum     : consec_down_days
      OBV          : obv, obv_zscore
    """
    df = df.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    # Moving averages
    for n in [5, 20, 50, 200]:
        df[f"ma_{n}"] = g["close"].rolling(n).mean().reset_index(level=0, drop=True)

    # RSI-2
    def _rsi(x):
        return RSIIndicator(close=x, window=rsi_period).rsi()
    df[f"rsi_{rsi_period}"] = g["close"].apply(_rsi).reset_index(level=0, drop=True)

    # Volume ratio
    df["vol_ma20"]  = g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # Returns
    df["ret_5d"]  = g["close"].pct_change(5) .reset_index(level=0, drop=True) * 100
    df["ret_10d"] = g["close"].pct_change(10).reset_index(level=0, drop=True) * 100
    df["ret_20d"] = g["close"].pct_change(20).reset_index(level=0, drop=True) * 100

    # ATR-14
    df["atr_14"] = g.apply(
        lambda x: (x["high"] - x["low"]).rolling(14).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    # Pct from MAs
    df["pct_from_ma200"] = (df["close"] - df["ma_200"]) / df["ma_200"] * 100
    df["pct_from_ma50"]  = (df["close"] - df["ma_50"])  / df["ma_50"]  * 100
    df["pct_from_ma20"]  = (df["close"] - df["ma_20"])  / df["ma_20"]  * 100

    # close_in_range: where did we close within today's range? (0=low, 1=high)
    day_range = df["high"] - df["low"]
    df["close_in_range"] = np.where(
        day_range > 0,
        (df["close"] - df["low"]) / day_range,
        0.5
    )

    # dist_52wk_low: % above lowest low of past 252 days
    df["low_252d"] = g["low"].rolling(252, min_periods=50).min().reset_index(level=0, drop=True)
    df["dist_52wk_low"] = (df["close"] - df["low_252d"]) / df["close"] * 100

    # consec_down_days: how many consecutive days has this stock been falling?
    def _consec_down(s):
        result = pd.Series(0.0, index=s.index)
        count  = 0
        for i in range(1, len(s)):
            count = count + 1 if s.iloc[i] < s.iloc[i-1] else 0
            result.iloc[i] = count
        return result

    df["consec_down_days"] = g["close"].apply(_consec_down).reset_index(level=0, drop=True)

    # OBV z-score (20-day rolling)
    def _obv(grp):
        closes = grp["close"].values
        vols   = grp["volume"].values
        obv_v  = np.zeros(len(grp))
        for i in range(1, len(grp)):
            if closes[i] > closes[i-1]:
                obv_v[i] = obv_v[i-1] + vols[i]
            elif closes[i] < closes[i-1]:
                obv_v[i] = obv_v[i-1] - vols[i]
            else:
                obv_v[i] = obv_v[i-1]
        return pd.Series(obv_v, index=grp.index)

    df["obv"]     = g.apply(_obv, include_groups=False).reset_index(level=0, drop=True)
    obv_mean      = g["obv"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    obv_std       = g["obv"].rolling(20, min_periods=5).std() .reset_index(level=0, drop=True)
    df["obv_zscore"] = (df["obv"] - obv_mean) / obv_std.replace(0, np.nan)

    return df


def add_spy_features(spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SPY-level features from a SPY-only bars dataframe.

    Input: date, close (+ optional open/high/low/volume)
    Adds: spy_ma200, spy_ma50, spy_rsi_14, spy_ret_5d, spy_ret_20d,
          spy_above_200, spy_above_50
    Returns a date-indexed dataframe suitable for merging into signal rows.
    """
    spy = spy_df.copy().sort_values("date").reset_index(drop=True)
    spy["spy_ma200"]     = spy["close"].rolling(200).mean()
    spy["spy_ma50"]      = spy["close"].rolling(50).mean()
    spy["spy_rsi_14"]    = RSIIndicator(close=spy["close"], window=14).rsi()
    spy["spy_ret_5d"]    = spy["close"].pct_change(5)  * 100
    spy["spy_ret_20d"]   = spy["close"].pct_change(20) * 100
    spy["spy_above_200"] = (spy["close"] > spy["spy_ma200"]).astype(float)
    spy["spy_above_50"]  = (spy["close"] > spy["spy_ma50"]).astype(float)
    return spy.set_index("date")


def assign_quintile(pred: float, thresholds: list) -> int:
    """
    Map a raw model prediction to a quintile (1-5) using saved percentile thresholds.
    thresholds: [p20, p40, p60, p80] from training data predictions.
    Q1 (worst) = below p20, Q5 (best) = above p80.
    """
    p20, p40, p60, p80 = thresholds
    if pred < p20:
        return 1
    if pred < p40:
        return 2
    if pred < p60:
        return 3
    if pred < p80:
        return 4
    return 5
