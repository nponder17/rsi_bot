"""
dashboard.py — Combined RSI Bot + Breakout Bot monitoring dashboard

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

DATABASE_URL    = os.getenv("DATABASE_URL", "").strip()
DB_PATH         = os.getenv("DB_PATH", "rsi_bot_state.sqlite")
ALPACA_KEY      = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET   = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")

ALPACA_HEADERS = {
    "APCA-API-KEY-ID"    : ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

# Table names per strategy
TABLES = {
    "RSI Bot"      : {"lots": "rsi_lots",         "gate": "rsi_llm_gate_log",  "events": "rsi_events"},
    "Breakout Bot" : {"lots": "bo_lots",           "gate": "bo_llm_gate_log",   "events": "bo_events"},
}

st.set_page_config(
    page_title = "Trading Bots Dashboard",
    page_icon  = "📈",
    layout     = "wide",
)

# ── DB helpers ────────────────────────────────────────────────────────────────

def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
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


def table_exists(table: str) -> bool:
    try:
        query_db(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


# ── Alpaca helpers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_alpaca_account() -> dict:
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=ALPACA_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def get_alpaca_positions() -> list[dict]:
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=ALPACA_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
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
    try:
        end  = (datetime.strptime(signal_date, "%Y-%m-%d") + timedelta(days=days + 5)).strftime("%Y-%m-%d")
        bars = get_bars(symbol, signal_date, end)
        if bars.empty or len(bars) < 2:
            return None
        entry_price = bars.iloc[0]["close"]
        exit_price  = bars.iloc[min(days, len(bars) - 1)]["close"]
        return (exit_price - entry_price) / entry_price
    except Exception:
        return None


# ── Data loaders ──────────────────────────────────────────────────────────────

def _numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=120)
def load_lots(strategy: str) -> pd.DataFrame:
    """Load lots for one strategy, or combined for 'Combined'."""
    numeric = ["filled_notional_entry", "filled_notional_exit", "qty",
               "avg_entry_price", "avg_exit_price", "notional"]

    if strategy == "Combined":
        frames = []
        for name, tbls in TABLES.items():
            tbl = tbls["lots"]
            df  = query_db(f"SELECT * FROM {tbl} ORDER BY lot_id")
            if not df.empty:
                df["strategy"] = name
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return _numeric_cols(combined, numeric)

    tbl = TABLES[strategy]["lots"]
    df  = query_db(f"SELECT * FROM {tbl} ORDER BY lot_id")
    if not df.empty:
        df["strategy"] = strategy
    return _numeric_cols(df, numeric)


@st.cache_data(ttl=120)
def load_gate_log(strategy: str) -> pd.DataFrame:
    if strategy == "Combined":
        frames = []
        for name, tbls in TABLES.items():
            df = query_db(f"SELECT * FROM {tbls['gate']} ORDER BY log_id")
            if not df.empty:
                df["strategy"] = name
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    tbl = TABLES[strategy]["gate"]
    df  = query_db(f"SELECT * FROM {tbl} ORDER BY log_id")
    if not df.empty:
        df["strategy"] = strategy
    return df


@st.cache_data(ttl=300)
def load_events(strategy: str, limit: int = 50) -> pd.DataFrame:
    tbl   = TABLES.get(strategy, TABLES["RSI Bot"])["events"]
    ph    = "%s" if DATABASE_URL else "?"
    return query_db(f"SELECT * FROM {tbl} ORDER BY ts DESC LIMIT {ph}", (limit,))


# ── Shared P&L computation ────────────────────────────────────────────────────

def compute_closed_pnl(closed: pd.DataFrame) -> pd.DataFrame:
    closed = closed.copy()
    closed["ret"]        = ((closed["filled_notional_exit"] - closed["filled_notional_entry"])
                            / closed["filled_notional_entry"])
    closed["pl_dollars"] = closed["filled_notional_exit"] - closed["filled_notional_entry"]
    closed["exit_date"]  = pd.to_datetime(closed["exit_date"])
    closed["entry_date"] = pd.to_datetime(closed["entry_date"])
    closed["hold_days"]  = (closed["exit_date"] - closed["entry_date"]).dt.days
    return closed.sort_values("exit_date")


# ── Page: Overview ────────────────────────────────────────────────────────────

def page_overview(strategy: str):
    st.title("📊 Overview")

    # ── Account metrics ───────────────────────────────────────────────────────
    acct = get_alpaca_account()
    if "error" in acct:
        st.warning(f"Alpaca: {acct['error']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity",       f"${float(acct.get('equity', 0)):,.2f}")
        c2.metric("Cash",         f"${float(acct.get('cash', 0)):,.2f}")
        c3.metric("Buying Power", f"${float(acct.get('buying_power', 0)):,.2f}")
        pl = float(acct.get("equity", 0)) - float(acct.get("last_equity", acct.get("equity", 0)))
        c4.metric("Today's P&L",  f"${pl:+,.2f}", delta=f"{pl:+.2f}")

    st.divider()

    # ── Strategy performance side by side ─────────────────────────────────────
    if strategy == "Combined":
        cols = st.columns(2)
        for i, (name, _) in enumerate(TABLES.items()):
            lots   = load_lots(name)
            closed = lots[lots["status"] == "CLOSED"].copy() if not lots.empty else pd.DataFrame()
            with cols[i]:
                st.subheader(f"{'📉' if name == 'RSI Bot' else '📈'} {name}")
                if not closed.empty and "filled_notional_entry" in closed.columns:
                    closed = compute_closed_pnl(closed)
                    wins      = (closed["ret"] > 0).sum()
                    win_rate  = wins / len(closed)
                    avg_ret   = closed["ret"].mean()
                    total_pl  = closed["pl_dollars"].sum()
                    c1, c2 = st.columns(2)
                    c1.metric("Trades",      f"{len(closed)}")
                    c2.metric("Win Rate",    f"{win_rate:.1%}")
                    c1.metric("Avg Return",  f"{avg_ret:+.2%}")
                    c2.metric("Realized P&L",f"${total_pl:+,.2f}")
                else:
                    st.info("No closed trades yet.")
    else:
        lots   = load_lots(strategy)
        closed = lots[lots["status"] == "CLOSED"].copy() if not lots.empty else pd.DataFrame()
        c1, c2, c3, c4 = st.columns(4)
        if not closed.empty and "filled_notional_entry" in closed.columns:
            closed   = compute_closed_pnl(closed)
            wins     = (closed["ret"] > 0).sum()
            win_rate = wins / len(closed)
            avg_ret  = closed["ret"].mean()
            total_pl = closed["pl_dollars"].sum()
            c1.metric("Total Trades",  f"{len(closed)}")
            c2.metric("Win Rate",      f"{win_rate:.1%}")
            c3.metric("Avg Return",    f"{avg_ret:+.2%}")
            c4.metric("Realized P&L",  f"${total_pl:+,.2f}")
        else:
            c1.metric("Total Trades", "0")
            c2.metric("Win Rate",     "—")
            c3.metric("Avg Return",   "—")
            c4.metric("Realized P&L", "—")

    st.divider()

    # ── Open positions + recent events ────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Open Positions")
        lots      = load_lots(strategy)
        open_df   = lots[lots["status"].isin(["OPEN", "PENDING_EXIT"])] if not lots.empty else pd.DataFrame()
        if open_df.empty:
            st.info("No open positions.")
        else:
            st.metric("Open", len(open_df))
            show_cols = ["symbol", "entry_date", "notional", "avg_entry_price"]
            if "strategy" in open_df.columns and strategy == "Combined":
                show_cols = ["strategy"] + show_cols
            if "exit_trigger" in open_df.columns:
                show_cols.append("exit_trigger")
            st.dataframe(open_df[[c for c in show_cols if c in open_df.columns]],
                         use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Last Bot Run")
        ev_strategy = list(TABLES.keys())[0] if strategy == "Combined" else strategy
        events = load_events(ev_strategy, limit=5)
        if events.empty:
            st.info("No events logged yet.")
        else:
            for _, ev in events.iterrows():
                st.caption(f"**{ev.get('event_type', '')}** — {str(ev.get('ts', ''))[:19]}")

    # ── Today's LLM gate ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Today's LLM Gate Decisions")
    gate = load_gate_log(strategy)
    if gate.empty:
        st.info("No gate decisions logged yet.")
    else:
        today_str  = str(date.today())
        today_gate = gate[gate["signal_date"] == today_str]
        if today_gate.empty:
            last_date = gate["signal_date"].max()
            st.caption(f"No decisions today — showing last run ({last_date})")
            today_gate = gate[gate["signal_date"] == last_date]

        for _, row in today_gate.iterrows():
            action = row.get("action", "")
            color  = "🔴" if action == "SKIP" else ("🔵" if action == "BOOST" else "🟢")
            strat_label = f" [{row.get('strategy','')}]" if strategy == "Combined" else ""
            with st.expander(f"{color} {row['symbol']}{strat_label} — {action}  |  {row.get('event_type','')}  |  conf={row.get('confidence', 0):.0%}"):
                st.write(f"**Reason:** {row.get('reason', '')}")
                if row.get("key_headline") and row["key_headline"] != "none":
                    st.write(f"**Headline:** {row.get('key_headline', '')}")
                c1, c2, c3 = st.columns(3)
                # RSI bot has rsi_2/ret_5d; breakout bot has model_score/ret_1d
                if "rsi_2" in row and pd.notna(row.get("rsi_2")):
                    c1.metric("RSI-2",      f"{row.get('rsi_2', 0):.1f}")
                    c2.metric("5d Change",  f"{row.get('ret_5d', 0):+.1f}%")
                elif "model_score" in row and pd.notna(row.get("model_score")):
                    c1.metric("Model Score", f"{row.get('model_score', 0):.3f}")
                    c2.metric("1d Change",   f"{row.get('ret_1d', 0):+.1%}")
                c3.metric("Articles", f"{int(row.get('n_articles', 0))}")


# ── Page: Open Positions ──────────────────────────────────────────────────────

def page_positions(strategy: str):
    st.title("📋 Open Positions")

    positions = get_alpaca_positions()
    lots      = load_lots(strategy)
    open_lots = lots[lots["status"].isin(["OPEN", "PENDING_EXIT"])] if not lots.empty else pd.DataFrame()

    if not positions:
        st.info("No open positions on Alpaca.")
        return

    total_mv  = sum(float(p.get("market_value", 0)) for p in positions)
    total_upl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
    winners   = sum(1 for p in positions if float(p.get("unrealized_pl", 0)) > 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions",      len(positions))
    c2.metric("Market Value",   f"${total_mv:,.2f}")
    c3.metric("Unrealized P&L", f"${total_upl:+,.2f}")
    c4.metric("Winners",        f"{winners}/{len(positions)}")

    st.divider()

    rows = []
    for p in positions:
        sym        = p.get("symbol", "")
        cur_price  = float(p.get("current_price", 0))
        avg_entry  = float(p.get("avg_entry_price", 0))
        mkt_val    = float(p.get("market_value", 0))
        unreal_pl  = float(p.get("unrealized_pl", 0))
        unreal_pct = float(p.get("unrealized_plpc", 0)) * 100

        entry_date   = "—"
        exit_trigger = "—"
        max_hold     = "—"
        strat_label  = "—"

        if not open_lots.empty:
            match = open_lots[open_lots["symbol"] == sym]
            if not match.empty:
                row = match.iloc[0]
                entry_date   = str(row.get("entry_date", "—"))
                exit_trigger = str(row.get("exit_trigger", "—")) if "exit_trigger" in row else "—"
                max_hold     = str(row.get("max_hold_date", "—")) if "max_hold_date" in row else "—"
                strat_label  = str(row.get("strategy", "—")) if "strategy" in row else "—"

        row_data = {
            "Symbol"     : sym,
            "Entry Date" : entry_date,
            "Entry Price": f"${avg_entry:.2f}",
            "Current"    : f"${cur_price:.2f}",
            "Mkt Value"  : f"${mkt_val:.2f}",
            "Unreal P&L" : f"${unreal_pl:+.2f}",
            "Return %"   : unreal_pct,
        }
        if strategy == "Combined":
            row_data["Strategy"] = strat_label
        if strategy in ("Breakout Bot", "Combined"):
            row_data["Exit Trigger"] = exit_trigger
            row_data["Max Hold"]     = max_hold

        rows.append(row_data)

    df = pd.DataFrame(rows)

    def color_ret(val):
        return "color: green" if val > 0 else "color: red"

    styled = df.style.map(color_ret, subset=["Return %"]).format({"Return %": "{:+.2f}%"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if len(df) > 0:
        st.divider()
        fig = px.bar(df, x="Symbol", y="Return %",
                     color="Return %",
                     color_continuous_scale=["red", "lightgray", "green"],
                     color_continuous_midpoint=0,
                     title="Unrealized Return by Position")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Trade History ───────────────────────────────────────────────────────

def page_history(strategy: str):
    st.title("📈 Trade History")

    lots = load_lots(strategy)
    if lots.empty:
        st.info("No trade history yet.")
        return

    closed = lots[lots["status"] == "CLOSED"].copy()
    if closed.empty:
        st.info("No closed trades yet.")
        return

    closed = compute_closed_pnl(closed)

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
    c1.metric("Avg Winner", f"{avg_win:+.2%}")
    c2.metric("Avg Loser",  f"{avg_loss:+.2%}")
    c3.metric("Winners",    f"{wins}")
    c4.metric("Losers",     f"{losses}")

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    if strategy == "Combined" and "strategy" in closed.columns:
        # One line per strategy + combined
        fig = go.Figure()
        colors = {"RSI Bot": "steelblue", "Breakout Bot": "darkorange"}
        for name in closed["strategy"].unique():
            sub = closed[closed["strategy"] == name].groupby("exit_date")["pl_dollars"].sum().reset_index()
            sub["cum"] = sub["pl_dollars"].cumsum()
            fig.add_trace(go.Scatter(x=sub["exit_date"], y=sub["cum"],
                                     mode="lines", name=name,
                                     line=dict(color=colors.get(name, "gray"), width=2)))
        # Combined
        all_daily = closed.groupby("exit_date")["pl_dollars"].sum().reset_index()
        all_daily["cum"] = all_daily["pl_dollars"].cumsum()
        fig.add_trace(go.Scatter(x=all_daily["exit_date"], y=all_daily["cum"],
                                 mode="lines", name="Combined",
                                 line=dict(color="green", width=2, dash="dot")))
    else:
        daily = closed.groupby("exit_date")["pl_dollars"].sum().reset_index()
        daily["cum"] = daily["pl_dollars"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["exit_date"], y=daily["cum"],
            mode="lines", fill="tozeroy",
            line=dict(color="green" if daily["cum"].iloc[-1] >= 0 else "red", width=2),
            name="Cumulative P&L",
        ))

    fig.update_layout(title="Equity Curve (Realized P&L)", xaxis_title="Date",
                      yaxis_title="Cumulative P&L ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly P&L ───────────────────────────────────────────────────────────
    closed["month"] = closed["exit_date"].dt.to_period("M").astype(str)

    if strategy == "Combined" and "strategy" in closed.columns:
        monthly = closed.groupby(["month", "strategy"])["pl_dollars"].sum().reset_index()
        fig2 = px.bar(monthly, x="month", y="pl_dollars", color="strategy",
                      title="Monthly P&L by Strategy ($)",
                      labels={"pl_dollars": "P&L ($)", "month": "Month"},
                      barmode="group",
                      color_discrete_map={"RSI Bot": "steelblue", "Breakout Bot": "darkorange"})
    else:
        monthly = closed.groupby("month")["pl_dollars"].sum().reset_index()
        monthly["color"] = monthly["pl_dollars"].apply(lambda x: "green" if x >= 0 else "red")
        fig2 = px.bar(monthly, x="month", y="pl_dollars", color="color",
                      color_discrete_map={"green": "green", "red": "red"},
                      title="Monthly P&L ($)", labels={"pl_dollars": "P&L ($)", "month": "Month"})
        fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Return distribution ────────────────────────────────────────────────────
    if strategy == "Combined" and "strategy" in closed.columns:
        fig3 = px.histogram(closed, x="ret", color="strategy", nbins=30,
                            title="Return Distribution by Strategy",
                            color_discrete_map={"RSI Bot": "steelblue", "Breakout Bot": "darkorange"},
                            barmode="overlay", opacity=0.7)
    else:
        fig3 = px.histogram(closed, x="ret", nbins=30,
                            title="Return Distribution",
                            color_discrete_sequence=["steelblue"])
    fig3.add_vline(x=0, line_dash="dash", line_color="red")
    fig3.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Exit trigger breakdown (breakout bot) ─────────────────────────────────
    if strategy in ("Breakout Bot", "Combined") and "exit_trigger" in closed.columns:
        st.divider()
        st.subheader("Exit Trigger Breakdown")
        trig_df = closed[closed["exit_trigger"].notna()].copy()
        if not trig_df.empty:
            col_left, col_right = st.columns(2)
            with col_left:
                counts = trig_df["exit_trigger"].value_counts().reset_index()
                counts.columns = ["Trigger", "Count"]
                fig_t = px.pie(counts, names="Trigger", values="Count",
                               title="Exit Reasons",
                               color="Trigger",
                               color_discrete_map={"TARGET": "green", "STOP": "red",
                                                   "TIME": "gray", "MANUAL": "steelblue"})
                st.plotly_chart(fig_t, use_container_width=True)
            with col_right:
                trig_ret = trig_df.groupby("exit_trigger")["ret"].agg(["mean","count"]).reset_index()
                trig_ret.columns = ["Trigger", "Avg Return", "Count"]
                fig_t2 = px.bar(trig_ret, x="Trigger", y="Avg Return",
                                color="Trigger",
                                color_discrete_map={"TARGET": "green", "STOP": "red",
                                                    "TIME": "gray", "MANUAL": "steelblue"},
                                title="Avg Return by Exit Trigger")
                fig_t2.update_yaxes(tickformat=".1%")
                fig_t2.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig_t2, use_container_width=True)

    # ── Trade table ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("All Closed Trades")

    base_cols = ["symbol", "entry_date", "exit_date", "hold_days",
                 "avg_entry_price", "avg_exit_price", "filled_notional_entry",
                 "pl_dollars", "ret"]
    if strategy == "Combined" and "strategy" in closed.columns:
        base_cols = ["strategy"] + base_cols
    if "exit_trigger" in closed.columns:
        base_cols.append("exit_trigger")
    if "model_score" in closed.columns:
        base_cols.append("model_score")
    if "llm_action" in closed.columns:
        base_cols.append("llm_action")

    display = closed[[c for c in base_cols if c in closed.columns]].rename(columns={
        "entry_date": "Entry", "exit_date": "Exit", "hold_days": "Days",
        "avg_entry_price": "Entry $", "avg_exit_price": "Exit $",
        "filled_notional_entry": "Notional", "pl_dollars": "P&L $", "ret": "Return",
        "exit_trigger": "Exit", "model_score": "Score", "llm_action": "LLM",
        "strategy": "Strategy",
    }).sort_values("Exit", ascending=False)

    def color_ret(val):
        return "color: green" if val > 0 else "color: red"

    fmt = {"Return": "{:+.2%}", "P&L $": "${:+.2f}",
           "Notional": "${:.0f}", "Entry $": "${:.2f}", "Exit $": "${:.2f}"}
    if "Score" in display.columns:
        fmt["Score"] = "{:.3f}"

    styled = display.style.map(color_ret, subset=["Return", "P&L $"]).format(fmt, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── Page: LLM Gate ────────────────────────────────────────────────────────────

def page_llm_gate(strategy: str):
    st.title("🤖 LLM Gate Log")

    gate = load_gate_log(strategy)
    if gate.empty:
        st.info("No LLM gate decisions logged yet.")
        return

    lots = load_lots(strategy)

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
    filter_cols = st.columns(3) if strategy == "Combined" else st.columns(2)
    with filter_cols[0]:
        action_filter = st.multiselect("Action", ["TAKE", "BOOST", "SKIP"],
                                       default=["TAKE", "BOOST", "SKIP"])
    with filter_cols[1]:
        event_types  = sorted(gate["event_type"].dropna().unique().tolist())
        event_filter = st.multiselect("Event Type", event_types, default=event_types)
    if strategy == "Combined" and "strategy" in gate.columns:
        with filter_cols[2]:
            strat_filter = st.multiselect("Strategy", list(TABLES.keys()),
                                          default=list(TABLES.keys()))
        filtered = gate[
            gate["action"].isin(action_filter) &
            gate["event_type"].isin(event_filter) &
            gate["strategy"].isin(strat_filter)
        ].copy()
    else:
        filtered = gate[
            gate["action"].isin(action_filter) &
            gate["event_type"].isin(event_filter)
        ].copy()

    # ── Join to trade outcomes ────────────────────────────────────────────────
    if not lots.empty and "entry_date" in lots.columns:
        closed = lots[lots["status"] == "CLOSED"].copy()
        if not closed.empty:
            closed["ret"] = ((closed["filled_notional_exit"] - closed["filled_notional_entry"])
                             / closed["filled_notional_entry"])
            lot_map = closed.set_index(["symbol", "entry_date"])["ret"].to_dict()
            filtered["actual_ret"] = filtered.apply(
                lambda r: lot_map.get((r["symbol"], r["signal_date"])), axis=1
            )
    else:
        filtered["actual_ret"] = None

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        action_counts = filtered["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig = px.pie(action_counts, names="Action", values="Count",
                     title="Gate Decisions",
                     color="Action",
                     color_discrete_map={"TAKE": "green", "BOOST": "steelblue", "SKIP": "red"})
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        event_counts = filtered["event_type"].value_counts().reset_index()
        event_counts.columns = ["Event Type", "Count"]
        fig2 = px.bar(event_counts, x="Count", y="Event Type", orientation="h",
                      title="Event Type Breakdown", color_discrete_sequence=["steelblue"])
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(filtered, x="confidence", color="action", nbins=10,
                        title="Confidence Distribution by Action",
                        color_discrete_map={"TAKE": "green", "BOOST": "steelblue", "SKIP": "red"},
                        barmode="overlay", opacity=0.7)
    fig3.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── SKIP outcome analysis ─────────────────────────────────────────────────
    st.subheader("🚫 SKIP Outcome Analysis")
    skipped = filtered[filtered["action"] == "SKIP"].copy()
    if skipped.empty:
        st.info("No SKIP decisions in current filter.")
    else:
        if st.button("Load SKIP outcomes (fetches price data)"):
            outcomes = []
            prog = st.progress(0)
            for i, (_, row) in enumerate(skipped.iterrows()):
                r5  = get_post_signal_return(row["symbol"], str(row["signal_date"]), 5)
                r10 = get_post_signal_return(row["symbol"], str(row["signal_date"]), 10)
                outcomes.append({
                    "symbol": row["symbol"], "signal_date": row["signal_date"],
                    "event_type": row["event_type"], "confidence": row["confidence"],
                    "5d_return": r5, "10d_return": r10,
                    "skip_correct_5d":  r5  < 0 if r5  is not None else None,
                    "skip_correct_10d": r10 < 0 if r10 is not None else None,
                })
                prog.progress((i + 1) / len(skipped))
            out_df = pd.DataFrame(outcomes)
            valid = out_df["skip_correct_5d"].notna().sum()
            c1, c2 = st.columns(2)
            c1.metric("SKIPs correct at 5d",  f"{out_df['skip_correct_5d'].sum()}/{valid}")
            c2.metric("SKIPs correct at 10d", f"{out_df['skip_correct_10d'].sum()}/{valid}")
            for col in ["5d_return", "10d_return"]:
                out_df[col] = out_df[col].apply(lambda v: f"{v:+.2%}" if v is not None else "—")
            out_df["✓ 5d"]  = out_df["skip_correct_5d"].map({True: "✅", False: "❌", None: "—"})
            out_df["✓ 10d"] = out_df["skip_correct_10d"].map({True: "✅", False: "❌", None: "—"})
            st.dataframe(out_df[["signal_date","symbol","event_type","confidence",
                                  "5d_return","✓ 5d","10d_return","✓ 10d"]],
                         use_container_width=True, hide_index=True)

    st.divider()

    # ── Full gate log ─────────────────────────────────────────────────────────
    st.subheader("Full Gate Log")
    display_cols = ["signal_date", "symbol", "action", "sentiment_score", "confidence",
                    "event_type", "reason", "key_headline", "n_articles", "n_filings"]
    # RSI-specific
    for col in ["rsi_2", "ret_5d"]:
        if col in filtered.columns:
            display_cols.append(col)
    # Breakout-specific
    for col in ["model_score", "ret_1d"]:
        if col in filtered.columns:
            display_cols.append(col)
    if strategy == "Combined" and "strategy" in filtered.columns:
        display_cols = ["strategy"] + display_cols
    if "actual_ret" in filtered.columns:
        display_cols.append("actual_ret")

    display_cols = [c for c in display_cols if c in filtered.columns]
    disp = filtered[display_cols].sort_values("signal_date", ascending=False).copy()

    if "actual_ret" in disp.columns:
        disp["actual_ret"] = disp["actual_ret"].apply(
            lambda x: f"{x:+.2%}" if pd.notna(x) else "open/—"
        )
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Page: Signal Quality ──────────────────────────────────────────────────────

def page_signals(strategy: str):
    st.title("🔬 Signal Quality")

    gate = load_gate_log(strategy)
    lots = load_lots(strategy)

    if lots.empty and gate.empty:
        st.info("Not enough data yet.")
        return

    # RSI-specific: RSI distribution
    if strategy != "Breakout Bot" and not gate.empty and "rsi_2" in gate.columns:
        st.subheader("RSI-2 at Signal")
        color_arg = {"color": "strategy"} if strategy == "Combined" else {"color_discrete_sequence": ["steelblue"]}
        fig = px.histogram(gate, x="rsi_2", nbins=20,
                           title="RSI-2 Distribution at Gate Input",
                           **color_arg)
        st.plotly_chart(fig, use_container_width=True)

    # Breakout-specific: model score distribution
    if strategy != "RSI Bot" and not gate.empty and "model_score" in gate.columns:
        st.subheader("Model Score at Signal")
        color_arg = {"color": "strategy"} if strategy == "Combined" else {"color_discrete_sequence": ["darkorange"]}
        fig = px.histogram(gate, x="model_score", nbins=20,
                           title="Model Score Distribution at Gate Input",
                           **color_arg)
        st.plotly_chart(fig, use_container_width=True)

    # Event type vs actual return
    if not lots.empty and not gate.empty:
        closed = lots[lots["status"] == "CLOSED"].copy()
        if not closed.empty:
            closed["ret"] = ((closed["filled_notional_exit"] - closed["filled_notional_entry"])
                             / closed["filled_notional_entry"])
            merged = gate.merge(
                closed[["symbol", "entry_date", "ret"]],
                left_on=["symbol", "signal_date"],
                right_on=["symbol", "entry_date"],
                how="inner"
            )
            if not merged.empty:
                st.subheader("Actual Return by Event Type")
                color_arg = {"color": "strategy"} if strategy == "Combined" and "strategy" in merged.columns else {}
                fig2 = px.box(merged, x="event_type", y="ret",
                              title="Return Distribution by Event Type",
                              **color_arg)
                fig2.update_yaxes(tickformat=".1%")
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Confidence Score Calibration")
                merged["conf_bucket"] = (merged["confidence"] * 10).round() / 10
                cal = merged.groupby("conf_bucket")["ret"].agg(["mean", "count"]).reset_index()
                cal.columns = ["Confidence", "Avg Return", "Count"]
                fig3 = px.scatter(cal, x="Confidence", y="Avg Return", size="Count",
                                  title="Does higher confidence = better returns?")
                fig3.update_xaxes(tickformat=".0%")
                fig3.update_yaxes(tickformat=".1%")
                fig3.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig3, use_container_width=True)

    if not gate.empty:
        st.subheader("Daily Signal Volume")
        daily_counts = gate.groupby(["signal_date", "action"]).size().reset_index(name="count")
        color_arg = {"color": "action",
                     "color_discrete_map": {"TAKE": "green", "BOOST": "steelblue", "SKIP": "red"}}
        fig4 = px.bar(daily_counts, x="signal_date", y="count",
                      title="Gate Decisions per Day", **color_arg)
        st.plotly_chart(fig4, use_container_width=True)


# ── Navigation ────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.title("📈 Trading Bots")
        st.caption("Personal monitoring dashboard")
        st.divider()

        strategy = st.radio(
            "Strategy",
            ["RSI Bot", "Breakout Bot", "Combined"],
            index=0,
        )

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
        page_overview(strategy)
    elif page == "Open Positions":
        page_positions(strategy)
    elif page == "Trade History":
        page_history(strategy)
    elif page == "LLM Gate Log":
        page_llm_gate(strategy)
    elif page == "Signal Quality":
        page_signals(strategy)


if __name__ == "__main__":
    main()
