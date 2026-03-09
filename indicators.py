import pandas as pd
from ta.momentum import RSIIndicator


def add_indicators(df: pd.DataFrame, rsi_period: int = 2) -> pd.DataFrame:
    """
    df expected columns: date, symbol, close
    Adds:
      - rsi_2
      - ma_5
      - ma_200
    """
    df = df.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    df["ma_5"] = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["ma_200"] = g["close"].rolling(200).mean().reset_index(level=0, drop=True)

    def rsi_apply(x: pd.Series) -> pd.Series:
        return RSIIndicator(close=x, window=rsi_period).rsi()

    df[f"rsi_{rsi_period}"] = g["close"].apply(rsi_apply).reset_index(level=0, drop=True)
    return df