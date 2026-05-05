"""
after_close.py — RSI-2 signal scan + ML v6 sizing + rv_5 vol filter + stop-loss

Run after market close each trading day. Produces a plan for the next open:
  - SELL: open positions where today's close > MA-20  (existing rule)
          OR close <= entry_price * (1 + STOP_LOSS_PCT/100)  (stop-loss)
  - BUY:  RSI-2 < 5, close > MA-200, rv_5 >= daily cross-section median
          → ML quintile notional sizing

Entry filter stack (in order):
  1. RSI-2 < 5               (oversold signal)
  2. close > MA-200          (long-term uptrend)
  3. close >= $2             (minimum price)
  4. rv_5 >= daily median    (top 50% most volatile stocks right now)
  5. not already open        (no re-entry while position held)

Exit triggers:
  - STOP_LOSS: close <= entry_price * 0.88 (12% drop)  [STOP_LOSS_PCT = -12]
  - MA20:      close > 20-day moving average           [MA_EXIT = 20]
  - Stop-loss takes priority if both fire same day.

ML v6 features (20 total) — currently disabled; flat NOTIONAL_PER_POSITION used.

Quintile sizing:  Q1=$80  Q2=$140  Q3=$200  Q4=$280  Q5=$400
Max new buys/day: 50
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

import pytz
import requests

from bot_config import (
    require_env, DATA_DIR, BOT_NAME,
    RSI_ENTRY_THRESHOLD, RSI_PERIOD,
    MIN_PRICE, USE_MA200_FILTER, MA_EXIT,
    MAX_NEW_BUYS_PER_DAY, MAX_TOTAL_OPEN_POSITIONS,
    NOTIONAL_PER_POSITION, QUINTILE_SIZE, MODEL_PATH,
    VIX_LOOKBACK_DAYS, VIX_CSV_URL, RV5_FILTER,
    LLM_GATE_ENABLED, LLM_GATE_MAX_CANDIDATES,
    STOP_LOSS_ENABLED, STOP_LOSS_PCT,
)
from llm_gate import run_llm_gate
from alpaca_utils import get_trading_calendar, get_next_trading_day, get_daily_bars
from indicators import add_indicators, add_spy_features, assign_quintile
from state_db import init_db, upsert_plan, log_event, log_llm_gate_decision, open_lots
from telegram_utils import tg_send

ET = pytz.timezone("America/New_York")

TEST_TODAY = None   # override for backtesting, e.g. "2024-01-05"

SPY_SYMBOL = "SPY"


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_vix() -> pd.DataFrame:
    """
    Download VIX history from CBOE (free, no key).
    Returns DataFrame with columns: date, vix_close, vix_ret_5d
    indexed by date.
    """
    try:
        resp = requests.get(VIX_CSV_URL, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"date": "date", "close": "vix_close"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        df["vix_ret_5d"] = df["vix_close"].pct_change(5) * 100
        return df.set_index("date")
    except Exception as e:
        print(f"  ⚠️  VIX fetch failed: {e}")
        return pd.DataFrame()


def pick_last_good_date(bars: pd.DataFrame, min_coverage: int):
    coverage    = bars.groupby("date")["symbol"].nunique().sort_index()
    last_date   = coverage.index.max()
    last_cov    = int(coverage.loc[last_date]) if last_date is not None else 0
    ok          = coverage[coverage >= min_coverage]
    last_good   = ok.index.max() if not ok.empty else None
    return last_good, coverage, last_date, last_cov


def score_candidate(row, model_payload, spy_row, vix_row) -> tuple:
    """
    Given a signal-date row for one symbol, extract all 20 features,
    run the ML model, and return (quintile, notional, pred_score).
    Falls back to Q3 / $200 if any feature is missing or model is None.
    """
    fallback = (3, NOTIONAL_PER_POSITION, None)

    if model_payload is None:
        return fallback

    features   = model_payload["features"]
    thresholds = model_payload["thresholds"]
    mdl        = model_payload["model"]
    q_size     = model_payload.get("quintile_size", QUINTILE_SIZE)

    feat_vals = {}

    # Per-symbol features (from signal-date bar)
    try:
        feat_vals["rsi_2_signal"]     = float(row[f"rsi_{RSI_PERIOD}"])
        feat_vals["pct_from_ma200"]   = float(row["pct_from_ma200"])
        feat_vals["pct_from_ma50"]    = float(row["pct_from_ma50"])
        feat_vals["pct_from_ma20"]    = float(row["pct_from_ma20"])
        feat_vals["vol_ratio"]        = float(row["vol_ratio"])
        feat_vals["ret_5d"]           = float(row["ret_5d"])
        feat_vals["ret_10d"]          = float(row["ret_10d"])
        feat_vals["ret_20d"]          = float(row["ret_20d"])
        feat_vals["atr_pct"]          = float(row["atr_pct"])
        feat_vals["close_in_range"]   = float(row["close_in_range"])
        feat_vals["dist_52wk_low"]    = float(row["dist_52wk_low"])
        feat_vals["consec_down_days"] = float(row["consec_down_days"])
        feat_vals["obv_zscore"]       = float(row["obv_zscore"])
    except (KeyError, TypeError, ValueError):
        return fallback

    # SPY features
    if spy_row is not None:
        try:
            feat_vals["spy_ret_5d"]    = float(spy_row["spy_ret_5d"])
            feat_vals["spy_ret_20d"]   = float(spy_row["spy_ret_20d"])
            feat_vals["spy_rsi_14"]    = float(spy_row["spy_rsi_14"])
            feat_vals["spy_above_200"] = float(spy_row["spy_above_200"])
            feat_vals["spy_above_50"]  = float(spy_row["spy_above_50"])
        except (KeyError, TypeError, ValueError):
            return fallback
    else:
        return fallback

    # VIX features
    if vix_row is not None:
        try:
            feat_vals["vix_close"]  = float(vix_row["vix_close"])
            feat_vals["vix_ret_5d"] = float(vix_row["vix_ret_5d"])
        except (KeyError, TypeError, ValueError):
            return fallback
    else:
        return fallback

    # Build feature vector in the exact order the model was trained on
    try:
        x = np.array([[feat_vals[f] for f in features]], dtype=float)
        if np.any(np.isnan(x)):
            return fallback
    except (KeyError, ValueError):
        return fallback

    pred     = float(mdl.predict(x)[0])
    quintile = assign_quintile(pred, thresholds)
    notional = q_size.get(quintile, NOTIONAL_PER_POSITION)
    return quintile, notional, pred


def main():
    require_env()
    init_db()
    ensure_dir()

    now_et = datetime.now(ET)
    today  = datetime.strptime(TEST_TODAY, "%Y-%m-%d").date() if TEST_TODAY else now_et.date()

    cal = get_trading_calendar(start=str(today - timedelta(days=10)),
                               end=str(today + timedelta(days=30)))
    if cal.empty:
        raise RuntimeError("Trading calendar empty; check Alpaca connectivity.")

    cal_dates = set(cal["date"].tolist())
    if today not in cal_dates:
        print(f"Not a trading day ({today}); exiting.")
        return

    next_td = get_next_trading_day(cal, today_date=today)


    # ── Fetch VIX ─────────────────────────────────────────────────────────────
    print("Fetching VIX from CBOE …")
    vix_idx = fetch_vix()
    if vix_idx.empty:
        print("  ⚠️  No VIX data — VIX gate disabled, VIX features will be missing.")

    # ── Load universe ─────────────────────────────────────────────────────────
    universe_path = os.path.join(DATA_DIR, "universe.csv")
    if not os.path.exists(universe_path):
        raise RuntimeError(f"Missing {universe_path}.")

    symbols = (pd.read_csv(universe_path)["symbol"]
               .dropna().astype(str).str.upper().unique().tolist())

    # Fetch bars for universe + SPY together (700 days for all indicators)
    fetch_symbols = list(set(symbols + [SPY_SYMBOL]))
    start = (today - timedelta(days=700)).isoformat()
    end   = (today + timedelta(days=1)).isoformat()

    print(f"Fetching bars for {len(fetch_symbols)} symbols (incl. SPY) …")
    all_bars = get_daily_bars(fetch_symbols, start=start, end=end)
    if all_bars.empty:
        raise RuntimeError("No bars returned.")

    all_bars["symbol"] = all_bars["symbol"].astype(str).str.upper()

    # Split SPY out before indicator calculation
    spy_bars  = all_bars[all_bars["symbol"] == SPY_SYMBOL].copy()
    univ_bars = all_bars[all_bars["symbol"].isin(set(symbols))].copy()

    # ── SPY features ──────────────────────────────────────────────────────────
    spy_feat_idx = pd.DataFrame()
    if not spy_bars.empty:
        spy_feat_idx = add_spy_features(spy_bars)
    else:
        print("  ⚠️  SPY bars not returned — SPY features will be missing.")

    # ── Universe indicators ───────────────────────────────────────────────────
    MIN_COVERAGE = min(150, max(30, int(0.6 * len(symbols))))
    last_good_date, coverage, last_date, last_cov = pick_last_good_date(univ_bars, MIN_COVERAGE)

    print("=== Data Diagnostics ===")
    print(f"Requested symbols: {len(symbols)}")
    print(f"Returned symbols:  {univ_bars['symbol'].nunique()}")
    print(f"Latest date:       {last_date} | coverage={last_cov}")
    print(f"Coverage thresh:   {MIN_COVERAGE}")
    print(f"Using date:        {last_good_date}")
    print("========================\n")

    if last_good_date is None:
        raise RuntimeError("No date met minimum coverage threshold.")

    # Filter to tradable price
    latest = (
        univ_bars[univ_bars["date"] == last_good_date][["symbol", "close"]]
        .rename(columns={"close": "last_close"})
    )
    tradable = latest[latest["last_close"] >= MIN_PRICE]["symbol"].tolist()
    df = univ_bars[univ_bars["symbol"].isin(tradable)].copy()

    print("Calculating indicators …")
    df = add_indicators(df, rsi_period=RSI_PERIOD)

    rsi_col  = f"rsi_{RSI_PERIOD}"
    exit_ma  = f"ma_{MA_EXIT}"

    # Signal-date slice
    day = df[df["date"] == last_good_date].copy()
    day = day.dropna(subset=[rsi_col, "ma_5"])
    day = day[day["close"] >= MIN_PRICE].copy()

    if USE_MA200_FILTER:
        day = day.dropna(subset=["ma_200"])
        day = day[day["close"] > day["ma_200"]].copy()

    # Entry candidates: RSI-2 < threshold
    entry_candidates = day[day[rsi_col] < RSI_ENTRY_THRESHOLD].copy()
    entry_candidates = entry_candidates.sort_values(rsi_col, ascending=True)

    # Current open positions
    current_open = open_lots(include_pending_entry=False)
    if current_open is None or current_open.empty:
        current_open = pd.DataFrame([])

    open_symbols = (
        set(current_open[current_open["status"].isin(["OPEN", "PENDING_EXIT"])]["symbol"]
            .unique().tolist())
        if not current_open.empty else set()
    )

    # Build per-symbol entry-price lookup for stop-loss check.
    # If a symbol has multiple lots open (rare — strategy disallows re-entry,
    # but defensive), use the WEIGHTED-AVERAGE entry price across qty.
    entry_price_by_symbol = {}
    if not current_open.empty:
        active = current_open[current_open["status"].isin(["OPEN", "PENDING_EXIT"])].copy()
        for sym, grp in active.groupby("symbol"):
            try:
                qtys   = grp["qty"].astype(float)
                prices = grp["avg_entry_price"].astype(float)
                total_qty = qtys.sum()
                if total_qty > 0:
                    entry_price_by_symbol[sym] = float((qtys * prices).sum() / total_qty)
            except (KeyError, ValueError, TypeError):
                continue   # skip if entry data malformed; lot will just not get stop check

    # ── Exit signals: stop-loss (priority) + MA-20 ────────────────────────────
    held_day = day[day["symbol"].isin(open_symbols)].copy()
    held_day = held_day.dropna(subset=[exit_ma])

    sell_reasons = {}   # symbol -> "STOP" or "MA20"

    # Stop-loss check first (priority)
    n_stop_triggers = 0
    if STOP_LOSS_ENABLED and entry_price_by_symbol:
        stop_mult = 1.0 + STOP_LOSS_PCT / 100.0   # e.g. 0.88 for -12%
        for _, hrow in held_day.iterrows():
            sym   = hrow["symbol"]
            close = hrow["close"]
            entry = entry_price_by_symbol.get(sym)
            if entry is None or pd.isna(close) or entry <= 0:
                continue
            if close <= entry * stop_mult:
                sell_reasons[sym] = "STOP"
                n_stop_triggers += 1

    # MA-20 check for everything not already flagged for stop
    for _, hrow in held_day.iterrows():
        sym = hrow["symbol"]
        if sym in sell_reasons:
            continue   # already flagged by stop
        if hrow["close"] > hrow[exit_ma]:
            sell_reasons[sym] = "MA20"

    sell_symbols = sorted(sell_reasons.keys())
    n_ma20_triggers = sum(1 for r in sell_reasons.values() if r == "MA20")

    # ── Entry signals with ML scoring ─────────────────────────────────────────
    buy_candidates = entry_candidates[~entry_candidates["symbol"].isin(open_symbols)].copy()

    open_count   = len(open_symbols)
    buy_capacity = max(0, MAX_TOTAL_OPEN_POSITIONS - open_count + len(sell_symbols))
    buy_capacity = min(buy_capacity, MAX_NEW_BUYS_PER_DAY)

    # Get SPY and VIX rows for the signal date
    spy_row = None
    if not spy_feat_idx.empty and last_good_date in spy_feat_idx.index:
        spy_row = spy_feat_idx.loc[last_good_date]

    vix_row = None
    vix_close_today = None
    vix_pct_today   = None
    if not vix_idx.empty and last_good_date in vix_idx.index:
        vix_row         = vix_idx.loc[last_good_date]
        vix_close_today = float(vix_row["vix_close"])
        # Rolling 252-day trailing percentile (no lookahead)
        past_vix = vix_idx["vix_close"].loc[:last_good_date].iloc[-(VIX_LOOKBACK_DAYS + 1):-1]
        if len(past_vix) >= 60:
            vix_pct_today = float(np.mean(past_vix.values < vix_close_today) * 100)

    # No ML model in use — all positions use flat notional size
    model_payload = None

    # ── rv_5 cross-sectional median filter ───────────────────────────────────
    rv5_filtered_count = 0
    if RV5_FILTER and "rv_5" in buy_candidates.columns:
        rv5_median = buy_candidates["rv_5"].median()
        pre_filter = len(buy_candidates)
        buy_candidates = buy_candidates[
            buy_candidates["rv_5"].notna() &
            (buy_candidates["rv_5"] >= rv5_median)
        ].copy()
        rv5_filtered_count = pre_filter - len(buy_candidates)
        print(f"  rv_5 filter: removed {rv5_filtered_count} low-vol candidates "
              f"(median rv_5={rv5_median:.4f}), {len(buy_candidates)} remain.")

    buy_symbols   = []
    buy_notionals = {}

    candidates_capped = buy_candidates.head(buy_capacity)

    # ── LLM signal gate ───────────────────────────────────────────────────────
    llm_skipped      = []
    llm_gated_count  = 0
    llm_gate_enabled = LLM_GATE_ENABLED

    if llm_gate_enabled:
        gate_input = candidates_capped.head(LLM_GATE_MAX_CANDIDATES)
        print(f"\nRunning LLM gate on {len(gate_input)} candidates …")
        try:
            candidates_capped, llm_skipped, analyses = run_llm_gate(
                candidates    = gate_input,
                signal_date   = str(last_good_date),
                rsi_col       = rsi_col,
                ret_col       = "ret_5d",
                verbose       = True,
            )
            llm_gated_count = len(llm_skipped)

            # Log every gate decision to DB for feedback loop analysis
            for sym, analysis in analyses.items():
                cand_row = gate_input[gate_input["symbol"] == sym]
                rsi_val  = float(cand_row[rsi_col].values[0]) if not cand_row.empty else 0.0
                ret_val  = float(cand_row["ret_5d"].values[0]) if not cand_row.empty else 0.0
                try:
                    log_llm_gate_decision(
                        signal_date     = str(last_good_date),
                        symbol          = sym,
                        rsi_2           = rsi_val,
                        ret_5d          = ret_val,
                        action          = analysis.action,
                        sentiment_score = analysis.sentiment_score,
                        confidence      = analysis.confidence,
                        event_type      = analysis.event_type,
                        reason          = analysis.reason,
                        key_headline    = analysis.key_headline,
                        n_articles      = getattr(analysis, "_n_articles", 0),
                        n_filings       = getattr(analysis, "_n_filings", 0),
                    )
                except Exception as log_err:
                    print(f"  ⚠️  Failed to log gate decision for {sym}: {log_err}")

        except Exception as e:
            print(f"  ⚠️  LLM gate failed: {e} — proceeding without gate")
            llm_gate_enabled = False
        print()

    for _, cand_row in candidates_capped.iterrows():
        sym      = cand_row["symbol"]
        quintile, notional, pred = score_candidate(cand_row, model_payload, spy_row, vix_row)
        buy_symbols.append(sym)
        buy_notionals[sym] = notional

    upsert_plan(
        plan_date=next_td,
        buy_symbols=buy_symbols,
        sell_symbols=sell_symbols,
        buy_notionals=buy_notionals,
    )

    # ── Telegram summary ──────────────────────────────────────────────────────
    vix_str   = (f"{vix_close_today:.1f} ({vix_pct_today:.0f}th pct)"
                 if vix_close_today and vix_pct_today is not None
                 else (f"{vix_close_today:.1f}" if vix_close_today else "N/A"))
    model_str = f"v6 ({model_payload.get('trained_on','?')})" if model_payload else "FLAT $200 (no model)"

    notional_summary = ""
    if buy_symbols and buy_notionals:
        q_dist = {}
        for sym in buy_symbols:
            n = buy_notionals.get(sym, NOTIONAL_PER_POSITION)
            q_dist[n] = q_dist.get(n, 0) + 1
        parts = [f"${k}×{v}" for k, v in sorted(q_dist.items())]
        notional_summary = f"  Notionals: {', '.join(parts)}"

    gate_str = "OFF"
    if llm_gate_enabled:
        gate_str = f"ON — skipped {llm_gated_count}"
        if llm_skipped:
            gate_str += f": {', '.join(s['symbol'] for s in llm_skipped)}"

    stop_str = "OFF"
    if STOP_LOSS_ENABLED:
        stop_str = f"ON @ {STOP_LOSS_PCT:+.0f}%  ({n_stop_triggers} triggered today)"

    # Sell breakdown by reason
    if sell_symbols:
        stop_syms = sorted([s for s, r in sell_reasons.items() if r == "STOP"])
        ma20_syms = sorted([s for s, r in sell_reasons.items() if r == "MA20"])
        sell_detail = []
        if stop_syms:
            sell_detail.append(f"🛑 STOP({len(stop_syms)}): {', '.join(stop_syms)}")
        if ma20_syms:
            sell_detail.append(f"📤 MA20({len(ma20_syms)}): {', '.join(ma20_syms)}")
        sell_str = "  " + "\n  ".join(sell_detail)
    else:
        sell_str = "  None"

    msg = [
        f"📊 {BOT_NAME} After-Close Plan  (v6 + rv_5 + LLM gate + stop-loss)",
        f"Signal date : {last_good_date}",
        f"Plan date   : {next_td}",
        f"Universe    : {len(tradable)} tradable (S&P 500)",
        f"VIX today   : {vix_str}",
        f"ML model    : {model_str}",
        f"Stop-loss   : {stop_str}",
        f"Open now    : {open_count}",
        f"Sell signals: {len(sell_symbols)} total",
        sell_str,
        f"Entry cands : {len(entry_candidates)} raw (RSI-2 < {RSI_ENTRY_THRESHOLD})",
        f"After rv_5  : {len(buy_candidates)} (removed {rv5_filtered_count} low-vol)",
        f"LLM gate    : {gate_str}",
        f"Buy capacity: {buy_capacity}",
        f"Buys planned: {len(buy_symbols)} → {', '.join(buy_symbols) if buy_symbols else 'None'}",
    ]
    if notional_summary:
        msg.append(notional_summary)

    # Append skip details if any (one line per skipped stock)
    if llm_skipped:
        msg.append("")
        msg.append("🚫 LLM gate skipped:")
        for s in llm_skipped:
            reason_short = s["reason"][:80] if s["reason"] else ""
            msg.append(f"  {s['symbol']}: [{s['event_type']}] {reason_short}")

    tg_send("\n".join(msg))
    log_event("AFTER_CLOSE", " | ".join(msg))
    print("\n".join(msg))


if __name__ == "__main__":
    main()