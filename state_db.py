import os
import pandas as pd

import psycopg
from psycopg.rows import dict_row

import sqlite3
from bot_config import DB_PATH

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# RSI bot table names (safe to share same Postgres instance with other bots)
LOTS_TABLE = "rsi_lots"
PLANNED_TABLE = "rsi_planned"
EVENTS_TABLE = "rsi_events"
EQUITY_TABLE = "rsi_equity_snapshots"


def _use_postgres() -> bool:
    return bool(DATABASE_URL)


def _pg_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def _sqlite_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {LOTS_TABLE} (
                    lot_id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT,
                    notional DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),

                    entry_client_order_id TEXT,
                    entry_order_id TEXT,
                    entry_filled_at TIMESTAMPTZ,
                    qty DOUBLE PRECISION,
                    avg_entry_price DOUBLE PRECISION,
                    filled_notional_entry DOUBLE PRECISION,

                    exit_client_order_id TEXT,
                    exit_order_id TEXT,
                    exit_filled_at TIMESTAMPTZ,
                    avg_exit_price DOUBLE PRECISION,
                    filled_notional_exit DOUBLE PRECISION,

                    fail_reason TEXT
                );
                """)

                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {PLANNED_TABLE} (
                    plan_date TEXT PRIMARY KEY,
                    buy_symbols TEXT,
                    sell_symbols TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    executed INTEGER DEFAULT 0,
                    executed_at TIMESTAMPTZ
                );
                """)

                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {EVENTS_TABLE} (
                    event_id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT NOW(),
                    event_type TEXT,
                    message TEXT
                );
                """)

                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {EQUITY_TABLE} (
                    snap_date TEXT PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT NOW(),
                    equity DOUBLE PRECISION,
                    cash DOUBLE PRECISION,
                    buying_power DOUBLE PRECISION,
                    bot_mv DOUBLE PRECISION,
                    bot_unrealized_pl DOUBLE PRECISION,
                    note TEXT
                );
                """)

                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{LOTS_TABLE}_status_symbol ON {LOTS_TABLE}(status, symbol);")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{LOTS_TABLE}_symbol_entry ON {LOTS_TABLE}(symbol, entry_date);")
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOTS_TABLE} (
            lot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT,
            notional REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            entry_client_order_id TEXT,
            entry_order_id TEXT,
            entry_filled_at TEXT,
            qty REAL,
            avg_entry_price REAL,
            filled_notional_entry REAL,
            exit_client_order_id TEXT,
            exit_order_id TEXT,
            exit_filled_at TEXT,
            avg_exit_price REAL,
            filled_notional_exit REAL,
            fail_reason TEXT
        );
        """)

        c.execute(f"""
        CREATE TABLE IF NOT EXISTS {PLANNED_TABLE} (
            plan_date TEXT PRIMARY KEY,
            buy_symbols TEXT,
            sell_symbols TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            executed INTEGER DEFAULT 0,
            executed_at TEXT
        );
        """)

        c.execute(f"""
        CREATE TABLE IF NOT EXISTS {EVENTS_TABLE} (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            event_type TEXT,
            message TEXT
        );
        """)

        c.execute(f"""
        CREATE TABLE IF NOT EXISTS {EQUITY_TABLE} (
            snap_date TEXT PRIMARY KEY,
            ts TEXT DEFAULT (datetime('now')),
            equity REAL,
            cash REAL,
            buying_power REAL,
            bot_mv REAL,
            bot_unrealized_pl REAL,
            note TEXT
        );
        """)

        c.execute(f"CREATE INDEX IF NOT EXISTS idx_{LOTS_TABLE}_status_symbol ON {LOTS_TABLE}(status, symbol);")
        c.execute(f"CREATE INDEX IF NOT EXISTS idx_{LOTS_TABLE}_symbol_entry ON {LOTS_TABLE}(symbol, entry_date);")
        c.commit()


def log_event(event_type: str, message: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"INSERT INTO {EVENTS_TABLE}(event_type, message) VALUES (%s, %s)", (event_type, message))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"INSERT INTO {EVENTS_TABLE}(event_type, message) VALUES (?, ?)", (event_type, message))
        c.commit()


def upsert_plan(plan_date: str, buy_symbols, sell_symbols):
    buys = ",".join(buy_symbols) if buy_symbols else ""
    sells = ",".join(sell_symbols) if sell_symbols else ""

    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                INSERT INTO {PLANNED_TABLE}(plan_date, buy_symbols, sell_symbols, executed, executed_at)
                VALUES (%s, %s, %s, 0, NULL)
                ON CONFLICT(plan_date) DO UPDATE SET
                    buy_symbols=EXCLUDED.buy_symbols,
                    sell_symbols=EXCLUDED.sell_symbols,
                    created_at=NOW(),
                    executed=0,
                    executed_at=NULL;
                """, (plan_date, buys, sells))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
        INSERT INTO {PLANNED_TABLE}(plan_date, buy_symbols, sell_symbols, executed, executed_at)
        VALUES (?, ?, ?, 0, NULL)
        ON CONFLICT(plan_date) DO UPDATE SET
            buy_symbols=excluded.buy_symbols,
            sell_symbols=excluded.sell_symbols,
            created_at=datetime('now'),
            executed=0,
            executed_at=NULL;
        """, (plan_date, buys, sells))
        c.commit()


def get_plan(plan_date: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                    SELECT plan_date, buy_symbols, sell_symbols, COALESCE(executed,0) AS executed
                    FROM {PLANNED_TABLE} WHERE plan_date=%s
                """, (plan_date,))
                row = cur.fetchone()
        if not row:
            return None
        return {
            "plan_date": row["plan_date"],
            "buy_symbols": [s for s in (row["buy_symbols"] or "").split(",") if s],
            "sell_symbols": [s for s in (row["sell_symbols"] or "").split(",") if s],
            "executed": bool(row["executed"]),
        }

    with _sqlite_conn() as c:
        cur = c.execute(f"""
            SELECT plan_date, buy_symbols, sell_symbols, COALESCE(executed,0)
            FROM {PLANNED_TABLE} WHERE plan_date=?
        """, (plan_date,))
        row = cur.fetchone()
    if not row:
        return None
    return {
        "plan_date": row[0],
        "buy_symbols": [s for s in (row[1] or "").split(",") if s],
        "sell_symbols": [s for s in (row[2] or "").split(",") if s],
        "executed": bool(row[3]),
    }


def plan_already_executed(plan_date: str) -> bool:
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"SELECT COALESCE(executed,0) AS executed FROM {PLANNED_TABLE} WHERE plan_date=%s", (plan_date,))
                row = cur.fetchone()
        return bool(row and int(row["executed"]) == 1)

    with _sqlite_conn() as c:
        cur = c.execute(f"SELECT COALESCE(executed,0) FROM {PLANNED_TABLE} WHERE plan_date=?", (plan_date,))
        row = cur.fetchone()
    return bool(row and int(row[0]) == 1)


def mark_plan_executed(plan_date: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                    UPDATE {PLANNED_TABLE}
                    SET executed=1, executed_at=NOW()
                    WHERE plan_date=%s
                """, (plan_date,))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
            UPDATE {PLANNED_TABLE}
            SET executed=1, executed_at=datetime('now')
            WHERE plan_date=?
        """, (plan_date,))
        c.commit()


def lot_exists_for_entry(symbol: str, entry_date: str) -> bool:
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                    SELECT 1 FROM {LOTS_TABLE}
                    WHERE symbol=%s AND entry_date=%s AND status IN ('PENDING_ENTRY','OPEN','PENDING_EXIT')
                    LIMIT 1
                """, (symbol, entry_date))
                return cur.fetchone() is not None

    with _sqlite_conn() as c:
        cur = c.execute(f"""
            SELECT 1 FROM {LOTS_TABLE}
            WHERE symbol=? AND entry_date=? AND status IN ('PENDING_ENTRY','OPEN','PENDING_EXIT')
            LIMIT 1
        """, (symbol, entry_date))
        return cur.fetchone() is not None


def add_lot_pending_entry(symbol: str, entry_date: str, notional: float, entry_client_order_id: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                INSERT INTO {LOTS_TABLE}(symbol, entry_date, exit_date, notional, status, entry_client_order_id)
                VALUES (%s, %s, NULL, %s, 'PENDING_ENTRY', %s)
                """, (symbol, entry_date, float(notional), entry_client_order_id))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
        INSERT INTO {LOTS_TABLE}(symbol, entry_date, exit_date, notional, status, entry_client_order_id)
        VALUES (?, ?, NULL, ?, 'PENDING_ENTRY', ?)
        """, (symbol, entry_date, float(notional), entry_client_order_id))
        c.commit()


def mark_lot_open_filled(entry_client_order_id: str, *, entry_order_id: str, qty: float, avg_entry_price: float, filled_notional: float, filled_at: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                UPDATE {LOTS_TABLE}
                SET status='OPEN',
                    entry_order_id=%s,
                    qty=%s,
                    avg_entry_price=%s,
                    filled_notional_entry=%s,
                    entry_filled_at=%s,
                    fail_reason=NULL
                WHERE entry_client_order_id=%s AND status='PENDING_ENTRY'
                """, (entry_order_id, float(qty), float(avg_entry_price), float(filled_notional), filled_at, entry_client_order_id))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
        UPDATE {LOTS_TABLE}
        SET status='OPEN',
            entry_order_id=?,
            qty=?,
            avg_entry_price=?,
            filled_notional_entry=?,
            entry_filled_at=?,
            fail_reason=NULL
        WHERE entry_client_order_id=? AND status='PENDING_ENTRY'
        """, (entry_order_id, float(qty), float(avg_entry_price), float(filled_notional), filled_at, entry_client_order_id))
        c.commit()


def mark_lot_failed(entry_client_order_id: str, reason: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                UPDATE {LOTS_TABLE}
                SET status='FAILED', fail_reason=%s
                WHERE entry_client_order_id=%s AND status='PENDING_ENTRY'
                """, (reason, entry_client_order_id))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
        UPDATE {LOTS_TABLE}
        SET status='FAILED', fail_reason=?
        WHERE entry_client_order_id=? AND status='PENDING_ENTRY'
        """, (reason, entry_client_order_id))
        c.commit()


def open_lots(include_pending_entry: bool = False) -> pd.DataFrame:
    statuses = ["OPEN", "PENDING_EXIT"]
    if include_pending_entry:
        statuses = ["PENDING_ENTRY"] + statuses

    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"SELECT * FROM {LOTS_TABLE} WHERE status = ANY(%s)", (statuses,))
                rows = cur.fetchall()
        return pd.DataFrame(rows)

    qmarks = ",".join(["?"] * len(statuses))
    with _sqlite_conn() as c:
        return pd.read_sql_query(f"SELECT * FROM {LOTS_TABLE} WHERE status IN ({qmarks})", c, params=tuple(statuses))


def get_open_lots_for_symbol(symbol: str) -> pd.DataFrame:
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                SELECT * FROM {LOTS_TABLE}
                WHERE symbol=%s AND status IN ('OPEN','PENDING_EXIT')
                ORDER BY lot_id ASC
                """, (symbol,))
                rows = cur.fetchall()
        return pd.DataFrame(rows)

    with _sqlite_conn() as c:
        return pd.read_sql_query(f"""
            SELECT * FROM {LOTS_TABLE}
            WHERE symbol=? AND status IN ('OPEN','PENDING_EXIT')
            ORDER BY lot_id ASC
        """, c, params=(symbol,))


def mark_lots_pending_exit(symbol: str, exit_client_order_id: str, exit_order_id: str):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                    UPDATE {LOTS_TABLE}
                    SET status='PENDING_EXIT',
                        exit_client_order_id=%s,
                        exit_order_id=%s
                    WHERE status='OPEN' AND symbol=%s
                """, (exit_client_order_id, exit_order_id, symbol))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
            UPDATE {LOTS_TABLE}
            SET status='PENDING_EXIT',
                exit_client_order_id=?,
                exit_order_id=?
            WHERE status='OPEN' AND symbol=?
        """, (exit_client_order_id, exit_order_id, symbol))
        c.commit()


def close_lots_for_symbol_filled(symbol: str, *, avg_exit_price: float, filled_notional_exit: float, filled_at: str, sold_qty_total: float | None = None, exit_date: str | None = None):
    lots = get_open_lots_for_symbol(symbol)
    if lots.empty:
        return

    qty_sum = float(lots["qty"].fillna(0.0).sum())
    use_qty = qty_sum > 0 and sold_qty_total is not None and sold_qty_total > 0

    notional_sum = float(lots["notional"].fillna(0.0).sum())
    if notional_sum <= 0:
        notional_sum = 1.0

    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                for _, r in lots.iterrows():
                    lot_id = int(r["lot_id"])
                    w = (float(r["qty"]) / qty_sum) if use_qty else (float(r["notional"]) / notional_sum)
                    alloc_qty = float(sold_qty_total) * w if (sold_qty_total is not None) else None
                    alloc_notional = float(filled_notional_exit) * w

                    cur.execute(f"""
                        UPDATE {LOTS_TABLE}
                        SET status='CLOSED',
                            exit_date=%s,
                            avg_exit_price=%s,
                            filled_notional_exit=%s,
                            exit_filled_at=%s,
                            qty=COALESCE(qty, %s)
                        WHERE lot_id=%s AND status IN ('OPEN','PENDING_EXIT')
                    """, (exit_date, float(avg_exit_price), alloc_notional, filled_at, alloc_qty, lot_id))
            c.commit()
        return

    with _sqlite_conn() as c:
        for _, r in lots.iterrows():
            lot_id = int(r["lot_id"])
            w = (float(r["qty"]) / qty_sum) if use_qty else (float(r["notional"]) / notional_sum)
            alloc_qty = float(sold_qty_total) * w if (sold_qty_total is not None) else None
            alloc_notional = float(filled_notional_exit) * w

            c.execute(f"""
                UPDATE {LOTS_TABLE}
                SET status='CLOSED',
                    exit_date=?,
                    avg_exit_price=?,
                    filled_notional_exit=?,
                    exit_filled_at=?,
                    qty=COALESCE(qty, ?)
                WHERE lot_id=? AND status IN ('OPEN','PENDING_EXIT')
            """, (exit_date, float(avg_exit_price), alloc_notional, filled_at, alloc_qty, lot_id))
        c.commit()


def get_pending_entries() -> pd.DataFrame:
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"SELECT * FROM {LOTS_TABLE} WHERE status='PENDING_ENTRY' ORDER BY lot_id ASC")
                rows = cur.fetchall()
        return pd.DataFrame(rows)

    with _sqlite_conn() as c:
        return pd.read_sql_query(f"SELECT * FROM {LOTS_TABLE} WHERE status='PENDING_ENTRY' ORDER BY lot_id ASC", c)


def get_pending_exits() -> pd.DataFrame:
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"SELECT * FROM {LOTS_TABLE} WHERE status='PENDING_EXIT' ORDER BY lot_id ASC")
                rows = cur.fetchall()
        return pd.DataFrame(rows)

    with _sqlite_conn() as c:
        return pd.read_sql_query(f"SELECT * FROM {LOTS_TABLE} WHERE status='PENDING_EXIT' ORDER BY lot_id ASC", c)


def upsert_equity_snapshot(snap_date: str, equity: float, cash: float, buying_power: float, bot_mv: float, bot_unrealized_pl: float, note: str = ""):
    if _use_postgres():
        with _pg_conn() as c:
            with c.cursor() as cur:
                cur.execute(f"""
                INSERT INTO {EQUITY_TABLE}(
                    snap_date, equity, cash, buying_power,
                    bot_mv, bot_unrealized_pl, note
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(snap_date) DO UPDATE SET
                    ts=NOW(),
                    equity=EXCLUDED.equity,
                    cash=EXCLUDED.cash,
                    buying_power=EXCLUDED.buying_power,
                    bot_mv=EXCLUDED.bot_mv,
                    bot_unrealized_pl=EXCLUDED.bot_unrealized_pl,
                    note=EXCLUDED.note;
                """, (snap_date, equity, cash, buying_power, bot_mv, bot_unrealized_pl, note))
            c.commit()
        return

    with _sqlite_conn() as c:
        c.execute(f"""
        INSERT INTO {EQUITY_TABLE}(
            snap_date, equity, cash, buying_power,
            bot_mv, bot_unrealized_pl, note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(snap_date) DO UPDATE SET
            ts=datetime('now'),
            equity=excluded.equity,
            cash=excluded.cash,
            buying_power=excluded.buying_power,
            bot_mv=excluded.bot_mv,
            bot_unrealized_pl=excluded.bot_unrealized_pl,
            note=excluded.note;
        """, (snap_date, equity, cash, buying_power, bot_mv, bot_unrealized_pl, note))
        c.commit()