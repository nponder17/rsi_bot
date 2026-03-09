import os
import pandas as pd
from datetime import datetime, timedelta
import pytz

from bot_config import (
    require_env, DATA_DIR, BOT_NAME,
    RSI_ENTRY_THRESHOLD, RSI_PERIOD,
    MIN_PRICE, USE_MA200_FILTER,
    MAX_NEW_BUYS_PER_DAY, MAX_TOTAL_OPEN_POSITIONS,
)
from alpaca_utils import get_trading_calendar, get_next_trading_day, get_daily_bars
from indicators import add_indicators
from state_db import init_db, upsert_plan, log_event, open_lots
from telegram_utils import tg_send

ET = pytz.timezone("America/New_York")

TEST_TODAY = None  # Friday


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def pick_last_good_date(bars: pd.DataFrame, min_coverage: int):
    coverage = bars.groupby("date")["symbol"].nunique().sort_index()
    last_date = coverage.index.max()
    last_cov = int(coverage.loc[last_date]) if last_date is not None else 0

    ok = coverage[coverage >= min_coverage]
    last_good = ok.index.max() if not ok.empty else None

    return last_good, coverage, last_date, last_cov


def main():
    require_env()
    init_db()
    ensure_dir()

    now_et = datetime.now(ET)
    today = datetime.strptime(TEST_TODAY, "%Y-%m-%d").date() if TEST_TODAY else now_et.date()

    cal = get_trading_calendar(start=str(today - timedelta(days=10)), end=str(today + timedelta(days=30)))
    if cal.empty:
        raise RuntimeError("Trading calendar empty; check Alpaca connectivity.")

    cal_dates = set(cal["date"].tolist())
    if today not in cal_dates:
        print(f"Not a trading day ({today}); exiting.")
        return

    next_td = get_next_trading_day(cal, today_date=today)

    universe_path = os.path.join(DATA_DIR, "universe.csv")
    if not os.path.exists(universe_path):
        raise RuntimeError(f"Missing {universe_path}. Create it with a 'symbol' column.")

    symbols = pd.read_csv(universe_path)["symbol"].dropna().astype(str).str.upper().unique().tolist()

    start = (today - timedelta(days=700)).isoformat()
    end = (today + timedelta(days=1)).isoformat()

    bars = get_daily_bars(symbols, start=start, end=end)
    if bars.empty:
        raise RuntimeError("No bars returned.")

    bars["symbol"] = bars["symbol"].astype(str).str.upper()

    MIN_COVERAGE = min(150, max(30, int(0.6 * len(symbols))))
    last_good_date, coverage, last_date, last_cov = pick_last_good_date(bars, MIN_COVERAGE)

    print("=== Data Diagnostics ===")
    print(f"Requested symbols: {len(symbols)}")
    print(f"Returned symbols:  {bars['symbol'].nunique()}")
    print(f"Latest date:       {last_date} | coverage={last_cov}")
    print(f"Coverage thresh:   {MIN_COVERAGE}")
    print(f"Using date:        {last_good_date}")
    print("========================\n")

    if last_good_date is None:
        raise RuntimeError("No date met minimum coverage threshold.")

    latest = (
        bars[bars["date"] == last_good_date][["symbol", "close"]]
        .rename(columns={"close": "last_close"})
    )
    tradable = latest[latest["last_close"] >= MIN_PRICE]["symbol"].tolist()

    df = bars[bars["symbol"].isin(tradable)].copy()
    df = add_indicators(df, rsi_period=RSI_PERIOD)

    rsi_col = f"rsi_{RSI_PERIOD}"

    day = df[df["date"] == last_good_date].copy()
    day = day.dropna(subset=[rsi_col, "ma_5"])
    day = day[day["close"] >= MIN_PRICE].copy()

    if USE_MA200_FILTER:
        day = day.dropna(subset=["ma_200"])
        day = day[day["close"] > day["ma_200"]].copy()

    entry_candidates = day[day[rsi_col] < RSI_ENTRY_THRESHOLD].copy()
    entry_candidates = entry_candidates.sort_values(rsi_col, ascending=True)

    current_open = open_lots(include_pending_entry=False)
    if current_open is None or current_open.empty:
        current_open = pd.DataFrame([])

    open_symbols = set(current_open[current_open["status"].isin(["OPEN", "PENDING_EXIT"])]["symbol"].unique().tolist()) if not current_open.empty else set()

    # Sells for tomorrow: currently open names whose latest close > ma_5
    held_day = day[day["symbol"].isin(open_symbols)].copy()
    sell_candidates = held_day[held_day["close"] > held_day["ma_5"]].copy()
    sell_symbols = sorted(sell_candidates["symbol"].unique().tolist())

    # Buys for tomorrow: new qualifying names not already open/pending
    buy_candidates = entry_candidates[~entry_candidates["symbol"].isin(open_symbols)].copy()

    open_count = len(open_symbols)
    buy_capacity = max(0, MAX_TOTAL_OPEN_POSITIONS - open_count + len(sell_symbols))
    buy_capacity = min(buy_capacity, MAX_NEW_BUYS_PER_DAY)

    buy_symbols = buy_candidates["symbol"].head(buy_capacity).tolist()

    upsert_plan(plan_date=next_td, buy_symbols=buy_symbols, sell_symbols=sell_symbols)

    msg = [
        f"📊 {BOT_NAME} After-Close Plan Created",
        f"Signal date used: {last_good_date}",
        f"Plan date (next open): {next_td}",
        f"Tradable universe (price>=${MIN_PRICE}): {len(tradable)}",
        f"Open symbols now: {open_count}",
        f"Sell signals for tomorrow ({len(sell_symbols)}): {', '.join(sell_symbols) if sell_symbols else 'None'}",
        f"Entry candidates ({len(entry_candidates)}): {len(entry_candidates)}",
        f"Buy capacity tomorrow: {buy_capacity}",
        f"Buys tomorrow ({len(buy_symbols)}): {', '.join(buy_symbols) if buy_symbols else 'None'}",
    ]

    tg_send("\n".join(msg))
    log_event("AFTER_CLOSE", " | ".join(msg))
    print("\n".join(msg))


if __name__ == "__main__":
    main()