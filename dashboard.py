"""
dashboard.py — RSI Bot personal monitoring dashboard

Local:    streamlit run dashboard.py
Railway:  streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timedelta, date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL      = os.getenv("DATABASE_URL", "").strip()
DB_PATH           = os.getenv("DB_PATH", "rsi_bot_state.sqlite")
ALPACA_KEY        = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET     = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL   = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")

ALPACA_HEADERS = {
    "APCA-API-KEY-ID"    : ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

st.set_page_config(
    page_title = "RSI Bot Dashboard",
    page_icon  = "📈",
    layout     = "wide",
)

# ── DB helpers ────────────────────────────────────────────────────────────────

def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame. Works with Postgres or SQLite."""
    try:
        if DATABASE_URL:
            import psycopg
            with psycopg.connect(DATABASE_URL) as conn:
                return pd.read_sql(sql, conn, params=params)
        else:
            with sqlite3.connect(DB_PATH) as conn:
                return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()


# ── Alpaca helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_alpaca_account() -> dict:
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/account",
                         headers=ALPACA_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def get_alpaca_positions() -> list[dict]:
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/positions",
                         headers=ALPACA_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily bars for a symbol between two dates (YYYY-MM-DD)."""
    try:
        r = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
            headers=ALPACA_HEADERS,
            params={"start": start, "end": end, "timeframe": "1Day",
                    "adjustment": "raw", "limit": 50},
            timeout=10,
        )
        r.raise_for_status()
        bars = r.json().get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"]).dt.date
        return df[["t", "c"]].rename(columns={"t": "date", "c": "close"})
    except Exception:
        return pd.DataFrame()


def get_post_signal_return(symbol: str, signal_date: str, days: int) -> float | None:
    """Return the stock's % return from signal_date to signal_date + days."""
    try:
        start = signal_date
        end   = (datetime.strptime(signal_date, "%Y-%m-%d") + timedelta(days=days + 5)).strftime("%Y-%m-%d")
        bars  = get_bars(symbol, start, end)
        if bars.empty or len(bars) < 2:
            return None
        entry_price = bars.iloc[0]["close"]
        target_idx  = min(days, len(bars) - 1)
        exit_price  = bars.iloc[target_idx]["close"]
        return (exit_price - entry_price) / entry_price
    except Exception:
        return None


# ── Shared data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_lots() -> pd.DataFrame:
    df = query_db("SELECT * FROM rsi_lots ORDER BY lot_id")
    if df.empty:
        return df
    for col in ["filled_notional_entry", "filled_notional_exit", "qty",
                 "avg_entry_price", "avg_exit_price", "notional"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=120)
def load_gate_log() -> pd.DataFrame:
    df = query_db("SELECT * FROM rsi_llm_gate_log ORDER BY log_id")
    if df.empty:
        return df
    for col in ["sentiment_score", "confidence", "rsi_2", "ret_5d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_events(limit: int = 50) -> pd.DataFrame:
    return query_db(
        "SELECT * FROM rsi_events ORDER BY ts DESC LIMIT %s" if DATABASE_URL
        else "SELECT * FROM rsi_events ORDER BY ts DESC LIMIT ?",
        (limit,)
    )


# ── Page: Overview ────────────────────────────────────────────────────────────

def page_overview():
    st.title("📊 Overview")

    # ── Account metrics from Alpaca ───────────────────────────────────────────
    acct = get_alpaca_account()
    if "error" in acct:
        st.warning(f"Alpaca connection issue: {acct['error']}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Equity",        f"${float(acct.get('equity',0)):,.2f}")
        col2.metric("Cash",          f"${float(acct.get('cash',0)):,.2f}")
        col3.metric("Buying Power",  f"${float(acct.get('buying_power',0)):,.2f}")
        pl = float(acct.get("equity", 0)) - float(acct.get("last_equity", acct.get("equity", 0)))
        col4.metric("Today's P&L",   f"${pl:+,.2f}", delta=f"{pl:+.2f}")

    st.divider()

    # ── Bot performance metrics ───────────────────────────────────────────────
    lots = load_lots()
    closed = lots[lots["status"] == "CLOSED"].copy() if not lots.empty else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)

    if not closed.empty and "filled_notional_entry" in closed.columns:
        closed["ret"] = (closed["filled_notional_exit"] - closed["filled_notional_entry"]) / closed["filled_notional_entry"]
        wins     = (closed["ret"] > 0).sum()
        win_rate = wins / len(closed) if len(closed) > 0 else 0
        avg_ret  = closed["ret"].mean()
        total_pl = (closed["filled_notional_exit"] - closed["filled_notional_entry"]).sum()

        col1.metric("Total Trades",  f"{len(closed)}")
        col2.metric("Win Rate",      f"{win_rate:.1%}")
        col3.metric("Avg Return",    f"{avg_ret:+.2%}")
        col4.metric("Realized P&L",  f"${total_pl:+,.2f}")
    else:
        col1.metric("Total Trades", "0")
        col2.metric("Win Rate",     "—")
        col3.metric("Avg Return",   "—")
        col4.metric("Realized P&L", "—")

    st.divider()

    # ── Open positions + LLM gate today ──────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Open Positions")
        open_lots = lots[lots["status"].isin(["OPEN", "PENDING_EXIT"])] if not lots.empty else pd.DataFrame()
        if open_lots.empty:
            st.info("No open positions.")
        else:
            st.metric("Open", len(open_lots))
            st.dataframe(
                open_lots[["symbol", "entry_date", "notional", "avg_entry_price"]].rename(columns={
                    "entry_date": "Entry Date", "notional": "Notional",
                    "avg_entry_price": "Entry Price"
                }),
                use_container_width=True, hide_index=True,
            )

    with col_right:
        st.subheader("Last Bot Run")
        events = load_events(limit=5)
        if events.empty:
            st.info("No events logged yet.")
        else:
            for _, ev in events.iterrows():
                st.caption(f"**{ev.get('event_type','')}** — {str(ev.get('ts',''))[:19]}")

    # ── LLM gate today ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Today's LLM Gate Decisions")
    gate = load_gate_log()
    if gate.empty:
        st.info("No gate decisions logged yet.")
    else:
        today_str = str(date.today())
        today_gate = gate[gate["signal_date"] == today_str]
        if today_gate.empty:
            last_date = gate["signal_date"].max()
            st.caption(f"No decisions today — showing last run ({last_date})")
            today_gate = gate[gate["signal_date"] == last_date]

        for _, row in today_gate.iterrows():
            action = row.get("action", "")
            color  = "🔴" if action == "SKIP" else ("🔵" if action == "BOOST" else "🟢")
            with st.expander(f"{color} {row['symbol']} — {action}  |  {row.get('event_type','')}  |  conf={row.get('confidence',0):.0%}"):
                st.write(f"**Reason:** {row.get('reason','')}")
                if row.get("key_headline") and row["key_headline"] != "none":
                    st.write(f"**Headline:** {row.get('key_headline','')}")
                c1, c2, c3 = st.columns(3)
                c1.metric("RSI-2",     f"{row.get('rsi_2',0):.1f}")
                c2.metric("5d Change", f"{row.get('ret_5d',0):+.1f}%")
                c3.metric("Articles",  f"{int(row.get('n_articles',0))}")


# ── Page: Open Positions ──────────────────────────────────────────────────────

def page_positions():
    st.title("📋 Open Positions")

    positions = get_alpaca_positions()
    lots      = load_lots()
    open_lots = lots[lots["status"].isin(["OPEN", "PENDING_EXIT"])] if not lots.empty else pd.DataFrame()

    if not positions:
        st.info("No open positions on Alpaca.")
        return

    rows = []
    for p in positions:
        sym        = p.get("symbol", "")
        qty        = float(p.get("qty", 0))
        cur_price  = float(p.get("current_price", 0))
        avg_entry  = float(p.get("avg_entry_price", 0))
        mkt_val    = float(p.get("market_value", 0))
        unreal_pl  = float(p.get("unrealized_pl", 0))
        unreal_pct = float(p.get("unrealized_plpc", 0)) * 100

        # Entry date from DB
        entry_date = "—"
        if not open_lots.empty:
            match = open_lots[open_lots["symbol"] == sym]
            if not match.empty:
                entry_date = str(match.iloc[0].get("entry_date", "—"))

        rows.append({
            "Symbol"     : sym,
            "Entry Date" : entry_date,
            "Qty"        : qty,
            "Entry Price": f"${avg_entry:.2f}",
            "Current"    : f"${cur_price:.2f}",
            "Mkt Value"  : f"${mkt_val:.2f}",
            "Unreal P&L" : f"${unreal_pl:+.2f}",
            "Return %"   : unreal_pct,
        })

    df = pd.DataFrame(rows)

    # Summary metrics
    total_mv  = sum(float(p.get("market_value", 0)) for p in positions)
    total_upl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
    winners   = sum(1 for p in positions if float(p.get("unrealized_pl", 0)) > 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions",     len(positions))
    c2.metric("Market Value",  f"${total_mv:,.2f}")
    c3.metric("Unrealized P&L",f"${total_upl:+,.2f}")
    c4.metric("Winners",       f"{winners}/{len(positions)}")

    st.divider()

    # Color-coded table
    def color_ret(val):
        color = "color: green" if val > 0 else ("color: red" if val < 0 else "")
        return color

    styled = df.style.map(color_ret, subset=["Return %"]).format({"Return %": "{:+.2f}%"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # P&L bar chart
    if len(df) > 0:
        st.divider()
        fig = px.bar(
            df, x="Symbol", y="Return %",
            color="Return %",
            color_continuous_scale=["red", "lightgray", "green"],
            color_continuous_midpoint=0,
            title="Unrealized Return by Position",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Trade History ───────────────────────────────────────────────────────

def page_history():
    st.title("📈 Trade History")

    lots = load_lots()
    if lots.empty:
        st.info("No trade history yet.")
        return

    closed = lots[lots["status"] == "CLOSED"].copy()
    if closed.empty:
        st.info("No closed trades yet — positions are still open.")
        return

    closed["ret"] = (
        (closed["filled_notional_exit"] - closed["filled_notional_entry"])
        / closed["filled_notional_entry"]
    )
    closed["pl_dollars"] = closed["filled_notional_exit"] - closed["filled_notional_entry"]
    closed["exit_date"]  = pd.to_datetime(closed["exit_date"])
    closed["entry_date"] = pd.to_datetime(closed["entry_date"])
    closed["hold_days"]  = (closed["exit_date"] - closed["entry_date"]).dt.days
    closed = closed.sort_values("exit_date")

    # ── Summary metrics ───────────────────────────────────────────────────────
    wins     = (closed["ret"] > 0).sum()
    losses   = (closed["ret"] <= 0).sum()
    win_rate = wins / len(closed)
    avg_win  = closed.loc[closed["ret"] > 0, "ret"].mean() if wins > 0 else 0
    avg_loss = closed.loc[closed["ret"] <= 0, "ret"].mean() if losses > 0 else 0
    total_pl = closed["pl_dollars"].sum()
    avg_hold = closed["hold_days"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades",  f"{len(closed)}")
    c2.metric("Win Rate",      f"{win_rate:.1%}")
    c3.metric("Realized P&L",  f"${total_pl:+,.2f}")
    c4.metric("Avg Hold",      f"{avg_hold:.1f} days")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Winner",   f"{avg_win:+.2%}")
    c2.metric("Avg Loser",    f"{avg_loss:+.2%}")
    c3.metric("Winners",      f"{wins}")
    c4.metric("Losers",       f"{losses}")

    st.divider()

    # ── Equity curve ─────────────────────────────────────────────────────────
    daily = closed.groupby("exit_date")["pl_dollars"].sum().reset_index()
    daily["cumulative_pl"] = daily["pl_dollars"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["exit_date"], y=daily["cumulative_pl"],
        mode="lines", fill="tozeroy",
        line=dict(color="green" if daily["cumulative_pl"].iloc[-1] >= 0 else "red", width=2),
        name="Cumulative P&L",
    ))
    fig.update_layout(title="Equity Curve (Realized P&L)", xaxis_title="Date",
                      yaxis_title="Cumulative P&L ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly P&L ───────────────────────────────────────────────────────────
    closed["month"] = closed["exit_date"].dt.to_period("M").astype(str)
    monthly = closed.groupby("month")["pl_dollars"].sum().reset_index()
    monthly["color"] = monthly["pl_dollars"].apply(lambda x: "green" if x >= 0 else "red")

    fig2 = px.bar(monthly, x="month", y="pl_dollars", color="color",
                  color_discrete_map={"green": "green", "red": "red"},
                  title="Monthly P&L ($)", labels={"pl_dollars": "P&L ($)", "month": "Month"})
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Return distribution ────────────────────────────────────────────────────
    fig3 = px.histogram(closed, x="ret", nbins=30,
                        title="Return Distribution",
                        labels={"ret": "Return"},
                        color_discrete_sequence=["steelblue"])
    fig3.add_vline(x=0, line_dash="dash", line_color="red")
    fig3.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Trade table ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("All Closed Trades")
    display = closed[[
        "symbol", "entry_date", "exit_date", "hold_days",
        "avg_entry_price", "avg_exit_price", "filled_notional_entry",
        "pl_dollars", "ret"
    ]].rename(columns={
        "entry_date": "Entry", "exit_date": "Exit", "hold_days": "Days",
        "avg_entry_price": "Entry $", "avg_exit_price": "Exit $",
        "filled_notional_entry": "Notional", "pl_dollars": "P&L $", "ret": "Return"
    }).sort_values("Exit", ascending=False)

    def color_ret(val):
        return "color: green" if val > 0 else "color: red"

    styled = display.style\
        .map(color_ret, subset=["Return", "P&L $"])\
        .format({"Return": "{:+.2%}", "P&L $": "${:+.2f}",
                 "Notional": "${:.0f}", "Entry $": "${:.2f}", "Exit $": "${:.2f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── Page: LLM Gate ────────────────────────────────────────────────────────────

def page_llm_gate():
    st.title("🤖 LLM Gate Log")

    gate = load_gate_log()
    if gate.empty:
        st.info("No LLM gate decisions logged yet.")
        return

    lots = load_lots()

    # ── Summary metrics ───────────────────────────────────────────────────────
    total  = len(gate)
    skips  = (gate["action"] == "SKIP").sum()
    takes  = (gate["action"] == "TAKE").sum()
    boosts = (gate["action"] == "BOOST").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Decisions", total)
    c2.metric("TAKE",  takes)
    c3.metric("BOOST", boosts)
    c4.metric("SKIP",  skips)

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        action_filter = st.multiselect("Action", ["TAKE", "BOOST", "SKIP"],
                                       default=["TAKE", "BOOST", "SKIP"])
    with col2:
        event_types = sorted(gate["event_type"].dropna().unique().tolist())
        event_filter = st.multiselect("Event Type", event_types, default=event_types)

    filtered = gate[
        gate["action"].isin(action_filter) &
        gate["event_type"].isin(event_filter)
    ].copy()

    # ── Outcome enrichment ────────────────────────────────────────────────────
    # Join TAKE/BOOST to actual trade results from rsi_lots
    if not lots.empty and "entry_date" in lots.columns:
        closed = lots[lots["status"] == "CLOSED"].copy()
        closed["ret"] = (
            (closed["filled_notional_exit"] - closed["filled_notional_entry"])
            / closed["filled_notional_entry"]
        )
        lot_map = closed.set_index(["symbol", "entry_date"])["ret"].to_dict()
        filtered["actual_ret"] = filtered.apply(
            lambda r: lot_map.get((r["symbol"], r["signal_date"])), axis=1
        )
    else:
        filtered["actual_ret"] = None

    st.divider()

    # ── Decision breakdown chart ──────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        action_counts = filtered["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig = px.pie(action_counts, names="Action", values="Count",
                     title="Gate Decisions",
                     color="Action",
                     color_discrete_map={"TAKE":"green","BOOST":"steelblue","SKIP":"red"})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        event_counts = filtered["event_type"].value_counts().reset_index()
        event_counts.columns = ["Event Type", "Count"]
        fig2 = px.bar(event_counts, x="Count", y="Event Type", orientation="h",
                      title="Event Type Breakdown", color_discrete_sequence=["steelblue"])
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Confidence distribution ───────────────────────────────────────────────
    fig3 = px.histogram(filtered, x="confidence", color="action", nbins=10,
                        title="Confidence Distribution by Action",
                        color_discrete_map={"TAKE":"green","BOOST":"steelblue","SKIP":"red"},
                        barmode="overlay", opacity=0.7)
    fig3.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── SKIP outcome analysis ─────────────────────────────────────────────────
    st.subheader("🚫 SKIP Outcome Analysis")
    st.caption("Did skipping actually avoid losses? Fetches post-signal prices from Alpaca.")

    skipped = filtered[filtered["action"] == "SKIP"].copy()
    if skipped.empty:
        st.info("No SKIP decisions in current filter.")
    else:
        if st.button("Load SKIP outcomes (fetches price data)"):
            outcomes = []
            prog = st.progress(0)
            for i, (_, row) in enumerate(skipped.iterrows()):
                r5  = get_post_signal_return(row["symbol"], str(row["signal_date"]), 5)
                r20 = get_post_signal_return(row["symbol"], str(row["signal_date"]), 20)
                outcomes.append({"symbol": row["symbol"],
                                  "signal_date": row["signal_date"],
                                  "event_type": row["event_type"],
                                  "confidence": row["confidence"],
                                  "5d_return": r5, "20d_return": r20,
                                  "skip_correct_5d": r5 < 0 if r5 is not None else None,
                                  "skip_correct_20d": r20 < 0 if r20 is not None else None})
                prog.progress((i + 1) / len(skipped))

            out_df = pd.DataFrame(outcomes)
            correct_5d  = out_df["skip_correct_5d"].sum()
            correct_20d = out_df["skip_correct_20d"].sum()
            valid       = out_df["skip_correct_5d"].notna().sum()

            c1, c2 = st.columns(2)
            c1.metric("SKIPs correct at 5d",  f"{correct_5d}/{valid}  ({correct_5d/max(valid,1):.0%})")
            c2.metric("SKIPs correct at 20d", f"{correct_20d}/{valid}  ({correct_20d/max(valid,1):.0%})")

            def fmt_pct(v):
                if v is None: return "—"
                return f"{v:+.2%}"
            out_df["5d_return"]  = out_df["5d_return"].apply(fmt_pct)
            out_df["20d_return"] = out_df["20d_return"].apply(fmt_pct)
            out_df["✓ 5d"] = out_df["skip_correct_5d"].map({True: "✅", False: "❌", None: "—"})
            out_df["✓ 20d"] = out_df["skip_correct_20d"].map({True: "✅", False: "❌", None: "—"})
            st.dataframe(out_df[["signal_date","symbol","event_type","confidence",
                                  "5d_return","✓ 5d","20d_return","✓ 20d"]],
                         use_container_width=True, hide_index=True)

    st.divider()

    # ── Full gate log table ───────────────────────────────────────────────────
    st.subheader("Full Gate Log")
    display_cols = ["signal_date","symbol","action","sentiment_score","confidence",
                    "event_type","reason","key_headline","n_articles","n_filings"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    if "actual_ret" in filtered.columns:
        display_cols.append("actual_ret")

    disp = filtered[display_cols].sort_values("signal_date", ascending=False).copy()

    if "actual_ret" in disp.columns:
        disp["actual_ret"] = disp["actual_ret"].apply(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "open/—"
        )

    st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Page: Signal Quality ──────────────────────────────────────────────────────

def page_signals():
    st.title("🔬 Signal Quality")

    gate = load_gate_log()
    lots = load_lots()

    if lots.empty and gate.empty:
        st.info("Not enough data yet.")
        return

    # ── RSI distribution at entry ─────────────────────────────────────────────
    if not gate.empty and "rsi_2" in gate.columns:
        st.subheader("RSI-2 at Signal")
        fig = px.histogram(gate, x="rsi_2", color="action", nbins=20,
                           title="RSI-2 Distribution at Gate Input",
                           color_discrete_map={"TAKE":"green","BOOST":"steelblue","SKIP":"red"},
                           barmode="overlay", opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)

    # ── Event type vs actual return ───────────────────────────────────────────
    if not lots.empty and not gate.empty:
        closed = lots[lots["status"] == "CLOSED"].copy()
        if not closed.empty:
            closed["ret"] = (
                (closed["filled_notional_exit"] - closed["filled_notional_entry"])
                / closed["filled_notional_entry"]
            )
            merged = gate.merge(
                closed[["symbol","entry_date","ret"]],
                left_on=["symbol","signal_date"],
                right_on=["symbol","entry_date"],
                how="inner"
            )
            if not merged.empty:
                st.subheader("Actual Return by Event Type")
                fig2 = px.box(merged, x="event_type", y="ret",
                              title="Return Distribution by Event Type",
                              color="event_type")
                fig2.update_yaxes(tickformat=".1%")
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)

                # ── Confidence calibration ────────────────────────────────────
                st.subheader("Confidence Score Calibration")
                merged["conf_bucket"] = (merged["confidence"] * 10).round() / 10
                cal = merged.groupby("conf_bucket")["ret"].agg(["mean","count"]).reset_index()
                cal.columns = ["Confidence", "Avg Return", "Count"]
                fig3 = px.scatter(cal, x="Confidence", y="Avg Return", size="Count",
                                  title="Does higher confidence = better returns?")
                fig3.update_xaxes(tickformat=".0%")
                fig3.update_yaxes(tickformat=".1%")
                fig3.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig3, use_container_width=True)

    # ── Daily signal count over time ──────────────────────────────────────────
    if not gate.empty:
        st.subheader("Daily Signal Volume")
        daily_counts = gate.groupby(["signal_date","action"]).size().reset_index(name="count")
        fig4 = px.bar(daily_counts, x="signal_date", y="count", color="action",
                      title="Gate Decisions per Day",
                      color_discrete_map={"TAKE":"green","BOOST":"steelblue","SKIP":"red"})
        st.plotly_chart(fig4, use_container_width=True)


# ── Navigation ────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.title("📈 RSI Bot")
        st.caption("Personal monitoring dashboard")
        st.divider()
        page = st.radio("Navigate", [
            "Overview",
            "Open Positions",
            "Trade History",
            "LLM Gate Log",
            "Signal Quality",
        ])
        st.divider()
        if st.button("🔄 Refresh data"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    if page == "Overview":
        page_overview()
    elif page == "Open Positions":
        page_positions()
    elif page == "Trade History":
        page_history()
    elif page == "LLM Gate Log":
        page_llm_gate()
    elif page == "Signal Quality":
        page_signals()


if __name__ == "__main__":
    main()
