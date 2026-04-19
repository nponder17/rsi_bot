"""LLM signal gate for the RSI bot.

Filters RSI-2 buy candidates through Claude to skip stocks with
fundamental damage (earnings misses, FDA rejections, fraud, etc.).

Public API:
    from llm_gate import run_llm_gate

    approved_df, skipped, analyses = run_llm_gate(
        candidates_df,
        signal_date="2024-08-01",
    )
"""

from __future__ import annotations

import time
import pandas as pd

from llm_gate.polygon_news import fetch_news, PolygonNewsError
from llm_gate.edgar import fetch_8k_filings, EdgarError
from llm_gate.analyzer import analyze_signal, SignalAnalysis

NEWS_DAYS_BACK    = 7
FILINGS_DAYS_BACK = 14
SLEEP_BETWEEN_API = 2   # seconds — respects Polygon free tier (5 req/min)


def run_llm_gate(
    candidates: pd.DataFrame,
    signal_date: str,
    rsi_col: str = "rsi_2",
    ret_col: str = "ret_5d",
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict], dict[str, SignalAnalysis]]:
    """Run the LLM gate on a DataFrame of RSI-2 buy candidates.

    Parameters
    ----------
    candidates : pd.DataFrame
        Each row is a buy candidate. Must have columns: symbol, <rsi_col>, <ret_col>.
    signal_date : str
        The as-of date (YYYY-MM-DD). Used to fetch historically accurate news.
    rsi_col : str
        Column name for RSI-2 values (default "rsi_2").
    ret_col : str
        Column name for recent price change in % (default "ret_5d").
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    approved : pd.DataFrame
        Subset of candidates that passed the gate (TAKE or BOOST).
    skipped : list[dict]
        List of {symbol, event_type, reason} for SKIP decisions.
    analyses : dict[str, SignalAnalysis]
        Full analysis object keyed by symbol, for all candidates.
    """
    approved_rows = []
    skipped       = []
    analyses      = {}

    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        symbol = str(row["symbol"])
        rsi    = float(row.get(rsi_col, 0.0))
        chg    = float(row.get(ret_col, 0.0))   # already in % from indicators.py

        if verbose:
            print(f"  LLM gate [{i}/{total}] {symbol}  RSI={rsi:.1f}  {ret_col}={chg:+.1f}% …",
                  flush=True)

        # Fetch news (capped at signal_date for historical accuracy)
        try:
            articles = fetch_news(
                symbol, days_back=NEWS_DAYS_BACK, limit=20, as_of_date=signal_date
            )
        except PolygonNewsError as e:
            if verbose:
                print(f"    [news error] {e}")
            articles = []

        time.sleep(SLEEP_BETWEEN_API)

        # Fetch 8-K filings
        try:
            filings = fetch_8k_filings(symbol, days_back=FILINGS_DAYS_BACK)
        except EdgarError as e:
            if verbose:
                print(f"    [edgar error] {e}")
            filings = []

        # Ask Claude
        result = analyze_signal(
            symbol           = symbol,
            rsi_2            = rsi,
            price_change_pct = chg,
            news_articles    = articles,
            filings          = filings,
            days_back        = NEWS_DAYS_BACK,
        )

        analyses[symbol] = result
        # Attach article/filing counts so after_close.py can log them
        result._n_articles = len(articles)
        result._n_filings  = len(filings)

        if result.skip_trade:
            skipped.append({
                "symbol"     : symbol,
                "event_type" : result.event_type,
                "reason"     : result.reason,
                "score"      : result.sentiment_score,
                "confidence" : result.confidence,
                "headline"   : result.key_headline,
            })
            if verbose:
                print(f"    → SKIP  score={result.sentiment_score:+.2f}  "
                      f"conf={result.confidence:.0%}  [{result.event_type}]")
                print(f"       {result.reason[:100]}")
        else:
            approved_rows.append(row)
            if verbose:
                action = "BOOST" if result.sentiment_score >= 0.5 else "TAKE"
                print(f"    → {action}  score={result.sentiment_score:+.2f}  "
                      f"conf={result.confidence:.0%}  [{result.event_type}]")

    approved = (
        pd.DataFrame(approved_rows).reset_index(drop=True)
        if approved_rows
        else candidates.iloc[0:0].copy()   # empty DataFrame with same columns
    )
    return approved, skipped, analyses
