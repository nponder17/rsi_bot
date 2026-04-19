from datetime import datetime, timedelta
import pytz

from bot_config import require_env, BOT_NAME, NOTIONAL_PER_POSITION, QUINTILE_SIZE
from alpaca_utils import (
    get_trading_calendar,
    submit_market_order,
    get_position,
    wait_for_order_terminal,
    get_order,
    get_order_by_client_order_id,
)
from state_db import (
    init_db, get_plan, plan_already_executed, mark_plan_executed,
    log_event,
    add_lot_pending_entry, mark_lot_open_filled, mark_lot_failed,
    mark_lots_pending_exit, close_lots_for_symbol_filled,
    lot_exists_for_entry, get_open_lots_for_symbol,
    get_pending_entries, get_pending_exits
)
from telegram_utils import tg_send

ET = pytz.timezone("America/New_York")

DRY_RUN = True
FORCE_EXEC_DATE = '2024-08-07'

MIN_SELL_QTY = 1e-6
FILL_TIMEOUT_SEC = 75
FILL_POLL_SEC = 1.5


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _order_terminal_summary(o: dict) -> str:
    st = (o.get("status") or "").lower()
    oid = o.get("id")
    coid = o.get("client_order_id")
    fq = o.get("filled_qty")
    fap = o.get("filled_avg_price")
    return f"status={st} id={oid} client_order_id={coid} filled_qty={fq} avg={fap}"


def _reconcile_pending():
    msgs = []

    pe = get_pending_entries()
    if pe is not None and not pe.empty:
        for _, r in pe.iterrows():
            sym = r["symbol"]
            coid = r.get("entry_client_order_id")
            oid = r.get("entry_order_id")

            try:
                if oid:
                    o = get_order(oid)
                elif coid:
                    o = get_order_by_client_order_id(coid)
                    if o is None:
                        msgs.append(f"⚠️ PENDING_ENTRY {sym}: order not found")
                        continue
                else:
                    msgs.append(f"⚠️ PENDING_ENTRY {sym}: missing order ids")
                    continue

                st = (o.get("status") or "").lower()
                if st == "filled":
                    filled_qty = _safe_float(o.get("filled_qty", 0.0))
                    avg_entry = _safe_float(o.get("filled_avg_price", 0.0))
                    filled_notional = filled_qty * avg_entry

                    mark_lot_open_filled(
                        coid,
                        entry_order_id=o.get("id") or oid or "UNKNOWN",
                        qty=filled_qty,
                        avg_entry_price=avg_entry,
                        filled_notional=filled_notional,
                        filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                    )
                    msgs.append(f"✅ Reconciled ENTRY fill: {sym} qty={filled_qty:.6f} avg=${avg_entry:.2f}")

                elif st in ("canceled", "rejected", "expired"):
                    mark_lot_failed(coid, f"reconciled_terminal:{st}")
                    msgs.append(f"🛑 ENTRY terminal {sym}: {st}")

            except Exception as e:
                msgs.append(f"❌ Reconcile PENDING_ENTRY {sym} failed: {e}")

    # Reconcile pending exits by symbol
    px = get_pending_exits()
    if px is not None and not px.empty:
        seen = set()
        for _, r in px.iterrows():
            sym = r["symbol"]
            if sym in seen:
                continue
            seen.add(sym)

            coid = r.get("exit_client_order_id")
            oid = r.get("exit_order_id")

            try:
                if oid:
                    o = get_order(oid)
                elif coid:
                    o = get_order_by_client_order_id(coid)
                    if o is None:
                        msgs.append(f"⚠️ PENDING_EXIT {sym}: order not found")
                        continue
                else:
                    msgs.append(f"⚠️ PENDING_EXIT {sym}: missing order ids")
                    continue

                st = (o.get("status") or "").lower()
                if st == "filled":
                    filled_qty = _safe_float(o.get("filled_qty", 0.0))
                    avg_exit = _safe_float(o.get("filled_avg_price", 0.0))
                    filled_notional_exit = filled_qty * avg_exit

                    close_lots_for_symbol_filled(
                        sym,
                        avg_exit_price=avg_exit,
                        filled_notional_exit=filled_notional_exit,
                        filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                        sold_qty_total=filled_qty,
                        exit_date=str(datetime.now(ET).date()),
                    )
                    msgs.append(f"✅ Reconciled EXIT fill: {sym} qty={filled_qty:.6f} avg=${avg_exit:.2f}")

            except Exception as e:
                msgs.append(f"❌ Reconcile PENDING_EXIT {sym} failed: {e}")

    return msgs


def main():
    require_env()
    init_db()

    now_et = datetime.now(ET)
    run_date = now_et.date()
    run_date_str = str(run_date)

    cal = get_trading_calendar(start=str(run_date - timedelta(days=10)), end=str(run_date + timedelta(days=60)))
    if cal.empty:
        raise RuntimeError("Trading calendar empty.")

    cal_dates = set(cal["date"].tolist())
    if (not FORCE_EXEC_DATE) and (run_date not in cal_dates):
        print(f"Not a trading day ({run_date}); exiting.")
        return

    exec_date = datetime.strptime(FORCE_EXEC_DATE, "%Y-%m-%d").date() if FORCE_EXEC_DATE else run_date
    exec_date_str = str(exec_date)

    rec_msgs = []
    if not DRY_RUN:
        rec_msgs = _reconcile_pending()

    plan = get_plan(exec_date_str)
    if plan is None:
        plan = {"buy_symbols": [], "sell_symbols": [], "executed": False, "plan_date": exec_date_str}
        log_event("AT_OPEN", f"No plan for {exec_date_str}; nothing to do except reconcile.")

    skip_buys = plan_already_executed(exec_date_str)

    # Exits first
    sell_msgs = []
    for sym in sorted(plan.get("sell_symbols", [])):
        try:
            lots = get_open_lots_for_symbol(sym)
            if lots is None or lots.empty:
                sell_msgs.append(f"⏭️ {sym}: no open lots found.")
                continue

            qty_to_sell = lots["qty"].apply(_safe_float).sum()
            if qty_to_sell <= MIN_SELL_QTY:
                sell_msgs.append(f"⏭️ {sym}: qty_to_sell≈0")
                continue

            pos = get_position(sym)
            if not pos:
                sell_msgs.append(f"⚠️ No Alpaca position for {sym}")
                continue

            broker_qty = abs(_safe_float(pos.get("qty", 0.0)))
            if broker_qty <= MIN_SELL_QTY:
                sell_msgs.append(f"⚠️ Alpaca qty ~0 for {sym}")
                continue

            qty_to_sell = min(qty_to_sell, broker_qty)
            exit_client_order_id = f"rsibot-{exec_date_str}-{sym}-sell"

            if DRY_RUN:
                sell_msgs.append(f"🧪 DRY_RUN would SELL {sym} qty={qty_to_sell:.6f}")
                continue

            resp = submit_market_order(
                symbol=sym,
                side="sell",
                qty=qty_to_sell,
                notional=None,
                time_in_force="day",
                client_order_id=exit_client_order_id,
            )
            exit_order_id = resp.get("id")

            mark_lots_pending_exit(sym, exit_client_order_id, exit_order_id)

            o = wait_for_order_terminal(
                order_id=exit_order_id,
                timeout_sec=FILL_TIMEOUT_SEC,
                poll_sec=FILL_POLL_SEC,
            )

            st = (o.get("status") or "").lower()
            if st != "filled":
                sell_msgs.append(f"⚠️ SELL not filled for {sym}: {_order_terminal_summary(o)}")
                continue

            filled_qty = _safe_float(o.get("filled_qty", 0.0))
            avg_exit = _safe_float(o.get("filled_avg_price", 0.0))
            filled_notional_exit = filled_qty * avg_exit

            close_lots_for_symbol_filled(
                sym,
                avg_exit_price=avg_exit,
                filled_notional_exit=filled_notional_exit,
                filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                sold_qty_total=filled_qty,
                exit_date=exec_date_str,
            )

            sell_msgs.append(f"✅ SELL filled: {sym} qty={filled_qty:.6f} avg=${avg_exit:.2f}")

        except Exception as e:
            sell_msgs.append(f"❌ SELL {sym} failed: {e}")

    if not sell_msgs:
        sell_msgs.append("No sells.")

    # Entries
    buy_msgs = []
    buy_success = 0

    if skip_buys:
        buy_msgs.append("No buys (plan already executed).")
    else:
        # Per-symbol ML notionals (from after_close.py ML scoring).
        # Falls back to NOTIONAL_PER_POSITION if not present.
        plan_notionals = plan.get("buy_notionals") or {}

        for sym in plan.get("buy_symbols", []):
            if lot_exists_for_entry(sym, exec_date_str):
                buy_msgs.append(f"⏭️ SKIP already have lot for {sym} entry={exec_date_str}")
                continue

            # Use ML-assigned notional; fall back to flat $200
            notional = float(plan_notionals.get(sym, NOTIONAL_PER_POSITION))

            entry_client_order_id = f"rsibot-{exec_date_str}-{sym}-buy"

            if DRY_RUN:
                buy_msgs.append(f"🧪 DRY_RUN would BUY {sym} notional=${notional:.2f}")
                continue

            try:
                add_lot_pending_entry(
                    symbol=sym,
                    entry_date=exec_date_str,
                    notional=notional,
                    entry_client_order_id=entry_client_order_id,
                )

                resp = submit_market_order(
                    symbol=sym,
                    side="buy",
                    notional=notional,
                    qty=None,
                    time_in_force="day",
                    client_order_id=entry_client_order_id,
                )
                entry_order_id = resp.get("id")

                o = wait_for_order_terminal(
                    order_id=entry_order_id,
                    timeout_sec=FILL_TIMEOUT_SEC,
                    poll_sec=FILL_POLL_SEC,
                )

                st = (o.get("status") or "").lower()
                if st != "filled":
                    buy_msgs.append(f"⚠️ BUY not filled for {sym}: {_order_terminal_summary(o)}")
                    mark_lot_failed(entry_client_order_id, f"entry_not_filled:{st}")
                    continue

                filled_qty = _safe_float(o.get("filled_qty", 0.0))
                avg_entry = _safe_float(o.get("filled_avg_price", 0.0))
                filled_notional_entry = filled_qty * avg_entry

                mark_lot_open_filled(
                    entry_client_order_id,
                    entry_order_id=o.get("id") or entry_order_id,
                    qty=filled_qty,
                    avg_entry_price=avg_entry,
                    filled_notional=filled_notional_entry,
                    filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                )

                buy_msgs.append(f"✅ BUY filled: {sym} notional=${notional:.0f} qty={filled_qty:.6f} avg=${avg_entry:.2f}")
                buy_success += 1

            except Exception as e:
                buy_msgs.append(f"❌ BUY {sym} failed: {e}")
                try:
                    mark_lot_failed(entry_client_order_id, f"exception:{e}")
                except Exception:
                    pass

    if (not DRY_RUN) and (not skip_buys):
        mark_plan_executed(exec_date_str)

    msg = [
        f"🚀 {BOT_NAME} At-Open Execution",
        f"Run date: {run_date_str}",
        f"Exec plan date: {exec_date_str}",
        f"Mode: {'DRY_RUN' if DRY_RUN else 'LIVE-PAPER'}",
        "",
    ]

    if rec_msgs:
        msg += ["Reconcile:"] + rec_msgs + [""]

    msg += ["Exits:"] + sell_msgs + [""]
    msg += ["Entries:"] + buy_msgs

    tg_send("\n".join(msg))
    log_event("AT_OPEN", " | ".join(msg))
    print("\n".join(msg))


if __name__ == "__main__":
    main()