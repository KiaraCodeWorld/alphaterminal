import time
import threading
from datetime import datetime
from bson import ObjectId
from .db import trades_collection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(val) -> float | None:
    try:
        return float(str(val).strip()) if val not in (None, "", "None") else None
    except (ValueError, TypeError):
        return None


def _to_int(val) -> int:
    try:
        return int(str(val).strip()) if val not in (None, "", "None") else 0
    except (ValueError, TypeError):
        return 0


def _fetch_price(symbol: str) -> float | None:
    try:
        import yfinance as yf
        info = yf.Ticker(symbol.replace(".", "-")).info or {}
        return info.get("regularMarketPrice") or info.get("currentPrice")
    except Exception:
        return None


def _calc_derived(doc: dict, current_price: float) -> dict:
    entry  = doc.get("entry_price") or 0
    qty    = doc.get("quantity") or 0
    side   = doc.get("side", "LONG")
    target = doc.get("target_price")
    sl     = doc.get("stop_loss")

    cost = entry * qty
    if side == "LONG":
        unrealized_pnl = (current_price - entry) * qty
    else:
        unrealized_pnl = (entry - current_price) * qty

    pnl_pct = (unrealized_pnl / cost * 100) if cost else 0.0

    prev_high = doc.get("highest_price_seen") or current_price
    prev_low  = doc.get("lowest_price_seen")  or current_price
    highest   = max(prev_high, current_price)
    lowest    = min(prev_low,  current_price)

    max_profit_pct = max_loss_pct = risk_reward = None
    if entry and target and sl:
        if side == "LONG":
            max_profit_pct = (target - entry) / entry * 100
            max_loss_pct   = (entry  - sl)    / entry * 100
        else:
            max_profit_pct = (entry  - target) / entry * 100
            max_loss_pct   = (sl     - entry)  / entry * 100
        if max_loss_pct and max_loss_pct > 0:
            risk_reward = max_profit_pct / max_loss_pct

    return {
        "cost":               round(cost, 2),
        "current_price":      round(current_price, 4),
        "unrealized_pnl":     round(unrealized_pnl, 2),
        "pnl_pct":            round(pnl_pct, 2),
        "highest_price_seen": round(highest, 4),
        "lowest_price_seen":  round(lowest,  4),
        "max_profit_pct":     round(max_profit_pct, 2) if max_profit_pct is not None else None,
        "max_loss_pct":       round(max_loss_pct, 2)   if max_loss_pct   is not None else None,
        "risk_reward":        round(risk_reward, 2)     if risk_reward    is not None else None,
        "updated_at":         datetime.now(),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

def add_trade(symbol: str, strategy_name: str, side: str, status: str,
              entry_price: str, quantity: str, target_price: str,
              stop_loss: str, notes: str, tags: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return "Symbol is required."
    ep = _to_float(entry_price)
    if not ep:
        return "Valid entry price is required."

    qty = _to_int(quantity)
    tp  = _to_float(target_price)
    sl  = _to_float(stop_loss)

    current_price = _fetch_price(symbol) or ep

    doc = {
        "symbol":             symbol,
        "strategy_name":      strategy_name.strip(),
        "side":               side.upper(),
        "status":             status.upper(),
        "entry_price":        ep,
        "entry_date":         datetime.now(),
        "quantity":           qty,
        "target_price":       tp,
        "stop_loss":          sl,
        "realized_pnl":       None,
        "exit_price":         None,
        "exit_date":          None,
        "notes":              notes.strip(),
        "tags":               [t.strip() for t in tags.split(",") if t.strip()],
        "created_at":         datetime.now(),
        "highest_price_seen": None,
        "lowest_price_seen":  None,
    }
    doc.update(_calc_derived(doc, current_price))
    trades_collection.insert_one(doc)
    return f"Trade logged: {side.upper()} {symbol} @ ${ep:,.2f} (qty {qty})"


def close_trade(trade_id: str, exit_price_str: str = "") -> str:
    ep = _to_float(exit_price_str)
    try:
        doc = trades_collection.find_one({"_id": ObjectId(trade_id)})
    except Exception:
        return "Invalid trade ID."
    if not doc:
        return "Trade not found."

    # If no price given, fetch current market price
    if not ep:
        ep = _fetch_price(doc.get("symbol", ""))
        if not ep:
            return "Could not fetch current price — enter manually."

    entry = doc.get("entry_price") or 0
    qty   = doc.get("quantity")   or 0
    side  = doc.get("side", "LONG")
    cost  = entry * qty

    realized_pnl = (ep - entry) * qty if side == "LONG" else (entry - ep) * qty
    pnl_pct      = (realized_pnl / cost * 100) if cost else 0.0

    trades_collection.update_one(
        {"_id": ObjectId(trade_id)},
        {"$set": {
            "status":        "CLOSED",
            "exit_price":    round(ep, 4),
            "exit_date":     datetime.now(),
            "realized_pnl":  round(realized_pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "current_price": round(ep, 4),
            "unrealized_pnl": 0.0,
            "updated_at":    datetime.now(),
        }},
    )
    return f"Closed @ ${ep:,.2f} · Realized P&L: ${realized_pnl:+,.2f} ({pnl_pct:+.1f}%)"


def delete_trade(trade_id: str) -> str:
    try:
        res = trades_collection.delete_one({"_id": ObjectId(trade_id)})
        return "Trade deleted." if res.deleted_count else "Trade not found."
    except Exception as exc:
        return f"Error: {exc}"


def update_trade_fields(trade_id: str, strategy: str, notes: str, tags: str) -> str:
    try:
        trades_collection.update_one(
            {"_id": ObjectId(trade_id)},
            {"$set": {
                "strategy_name": strategy.strip(),
                "notes":         notes.strip(),
                "tags":          [t.strip() for t in tags.split(",") if t.strip()],
                "updated_at":    datetime.now(),
            }},
        )
        return "Trade updated successfully."
    except Exception as exc:
        return f"Error: {exc}"


def load_trade_for_edit(trade_id: str) -> tuple[str, str, str, str]:
    try:
        doc = trades_collection.find_one({"_id": ObjectId(trade_id.strip())})
        if not doc:
            return "", "", "", "Trade not found."
        tags_str = ", ".join(doc.get("tags") or [])
        label    = f"{doc['symbol']} {doc['side']} @ ${doc['entry_price']:,.2f}"
        return (
            doc.get("strategy_name", ""),
            doc.get("notes", ""),
            tags_str,
            f"Editing **{label}** — update fields then click Save.",
        )
    except Exception as exc:
        return "", "", "", f"Error: {exc}"


# ── Price refresh ─────────────────────────────────────────────────────────────

def refresh_open_prices() -> str:
    open_docs = list(trades_collection.find({"status": "OPEN"}))
    if not open_docs:
        return "No open trades to refresh."
    updated = 0
    for doc in open_docs:
        price = _fetch_price(doc["symbol"])
        if not price:
            continue
        trades_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": _calc_derived(doc, price)},
        )
        updated += 1
    return f"Refreshed {updated}/{len(open_docs)} open trade prices."


def _bg_refresh_loop() -> None:
    while True:
        time.sleep(600)
        try:
            refresh_open_prices()
        except Exception:
            pass


threading.Thread(target=_bg_refresh_loop, daemon=True).start()


# ── Query ─────────────────────────────────────────────────────────────────────

_SORT_MAP = {
    "entry_date": ("entry_date", -1),
    "pnl_pct":    ("pnl_pct",   -1),
    "symbol":     ("symbol",     1),
    "status":     ("status",     1),
}


def get_trades(status_filter: str = "ALL", sort_by: str = "entry_date") -> list[dict]:
    query = {} if status_filter == "ALL" else {"status": status_filter}
    field, direction = _SORT_MAP.get(sort_by, ("entry_date", -1))
    return list(trades_collection.find(query).sort(field, direction))


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_trades_html(status_filter: str = "ALL", sort_by: str = "entry_date") -> str:
    trades = get_trades(status_filter, sort_by)

    if not trades:
        return "<p style='color:#94a3b8; padding:18px;'>No trades logged yet. Use the form above to log your first trade.</p>"

    open_count   = sum(1 for t in trades if t.get("status") == "OPEN")
    closed_count = sum(1 for t in trades if t.get("status") == "CLOSED")
    open_pnl     = sum(t.get("unrealized_pnl") or 0 for t in trades if t.get("status") == "OPEN")
    realized_pnl = sum(t.get("realized_pnl")   or 0 for t in trades if t.get("status") == "CLOSED")
    total_pnl    = open_pnl + realized_pnl
    total_c      = "#00d68f" if total_pnl >= 0 else "#f04040"
    open_c       = "#00d68f" if open_pnl  >= 0 else "#f04040"
    real_c       = "#00d68f" if realized_pnl >= 0 else "#f04040"

    summary = (
        f"<div style='display:flex; gap:20px; margin-bottom:14px; flex-wrap:wrap;'>"
        f"<span style='color:#4a5568; font-size:0.78rem;'>OPEN <b style='color:#e2e8f4;'>{open_count}</b></span>"
        f"<span style='color:#4a5568; font-size:0.78rem;'>CLOSED <b style='color:#e2e8f4;'>{closed_count}</b></span>"
        f"<span style='color:#4a5568; font-size:0.78rem;'>UNREALIZED P&amp;L "
        f"<b style='color:{open_c};'>${open_pnl:+,.2f}</b></span>"
        f"<span style='color:#4a5568; font-size:0.78rem;'>REALIZED P&amp;L "
        f"<b style='color:{real_c};'>${realized_pnl:+,.2f}</b></span>"
        f"<span style='color:#4a5568; font-size:0.78rem;'>TOTAL "
        f"<b style='color:{total_c};'>${total_pnl:+,.2f}</b></span>"
        f"</div>"
    )

    rows = []
    for doc in trades:
        tid    = str(doc["_id"])
        sym    = doc.get("symbol", "")
        side   = doc.get("side", "LONG")
        status = doc.get("status", "OPEN")
        strat  = (doc.get("strategy_name") or "—")[:22]
        entry  = doc.get("entry_price") or 0
        qty    = doc.get("quantity") or 0
        cur    = doc.get("current_price") or entry
        target = doc.get("target_price")
        sl     = doc.get("stop_loss")
        rr     = doc.get("risk_reward")
        tags   = doc.get("tags") or []
        notes  = doc.get("notes") or ""
        upd    = doc.get("updated_at") or doc.get("created_at")

        # P&L values
        if status == "CLOSED":
            pnl_val = doc.get("realized_pnl") or 0
        else:
            pnl_val = doc.get("unrealized_pnl") or 0
        pnl_pct = doc.get("pnl_pct") or 0

        pnl_fg = "#00d68f" if pnl_val >= 0 else "#f04040"
        pnl_bg = "#0d2b1a" if pnl_val >= 0 else "#2b0d0d"

        # Side badge
        side_c     = "#00d68f" if side == "LONG" else "#f87171"
        side_badge = (
            f"<span style='color:{side_c}; font-weight:700; font-size:0.78rem; "
            f"letter-spacing:0.04em;'>{side}</span>"
        )

        # Status badge
        if status == "OPEN":
            status_badge = (
                "<span style='background:#0d2b1a; color:#00d68f; border:1px solid #00d68f44; "
                "border-radius:12px; padding:2px 9px; font-size:0.7rem; font-weight:700;'>OPEN</span>"
            )
        else:
            status_badge = (
                "<span style='background:#161b22; color:#7a8499; border:1px solid #2a3347; "
                "border-radius:12px; padding:2px 9px; font-size:0.7rem; font-weight:700;'>CLOSED</span>"
            )

        # P&L cell
        pnl_cell = (
            f"<span style='background:{pnl_bg}; color:{pnl_fg}; border-radius:6px; "
            f"padding:4px 9px; font-weight:700; font-size:0.82rem; display:inline-block; "
            f"min-width:78px; text-align:right; line-height:1.5;'>"
            f"${pnl_val:+,.2f}<br>"
            f"<span style='font-size:0.7rem;'>{pnl_pct:+.1f}%</span></span>"
        )

        # Levels
        target_str = f"${target:,.2f}" if target else "—"
        sl_str     = f"${sl:,.2f}"     if sl     else "—"
        rr_str     = f"{rr:.1f}x"      if rr     else "—"

        # Tag chips
        tag_chips = "".join(
            f"<span style='background:#1e2536; color:#7a8499; border-radius:4px; "
            f"padding:1px 5px; font-size:0.66rem; margin:1px 1px 0 0;'>{t}</span>"
            for t in tags[:4]
        )

        upd_str    = upd.strftime("%m/%d %H:%M") if upd else "—"
        notes_trim = notes[:50] + ("…" if len(notes) > 50 else "")

        tv_url = f"https://www.tradingview.com/symbols/{sym.replace('.', '-')}/"
        sym_cell = (
            f"<a href='{tv_url}' target='_blank' "
            f"style='color:#00d68f; font-weight:700; text-decoration:none; font-size:0.9rem;'>{sym}</a>"
            + (f"<br><span style='color:#4a5568; font-size:0.68rem;'>{tag_chips}</span>" if tag_chips else "")
        )

        close_btn = (
            f"<button onclick=\"closeTrade('{tid}')\" "
            f"style='background:#0d2b1a; color:#00d68f; border:1px solid #00d68f44; "
            f"border-radius:5px; padding:3px 8px; font-size:0.7rem; cursor:pointer; "
            f"font-weight:600; margin:1px; white-space:nowrap;'>✓ Close</button>"
        ) if status == "OPEN" else ""

        edit_btn = (
            f"<button onclick=\"editTrade('{tid}')\" "
            f"style='background:#1e2536; color:#94a3b8; border:1px solid #2a3347; "
            f"border-radius:5px; padding:3px 8px; font-size:0.7rem; cursor:pointer; "
            f"font-weight:600; margin:1px; white-space:nowrap;'>✎ Edit</button>"
        )

        del_btn = (
            f"<button onclick=\"deleteTrade('{tid}')\" "
            f"style='background:#2b0d0d; color:#f04040; border:1px solid #f0404044; "
            f"border-radius:5px; padding:3px 8px; font-size:0.7rem; cursor:pointer; "
            f"font-weight:600; margin:1px;'>✕</button>"
        )

        rows.append(
            f"<tr>"
            f"<td>{sym_cell}</td>"
            f"<td>{side_badge}</td>"
            f"<td>{status_badge}</td>"
            f"<td style='color:#e2e8f4; font-weight:600;'>${entry:,.2f}"
            f"<br><span style='color:#4a5568; font-size:0.7rem;'>qty {qty}</span></td>"
            f"<td style='color:#e2e8f4;'>${cur:,.2f}</td>"
            f"<td>{pnl_cell}</td>"
            f"<td style='color:#7a8499; font-size:0.8rem;'>{target_str}"
            f"<br><span style='color:#f04040; font-size:0.75rem;'>{sl_str}</span></td>"
            f"<td style='color:#7a8499; font-size:0.82rem;'>{rr_str}</td>"
            f"<td style='color:#7a8499; font-size:0.8rem;'>{strat}</td>"
            f"<td style='color:#7a8499; font-size:0.77rem; max-width:160px;'>{notes_trim}</td>"
            f"<td style='color:#4a5568; font-size:0.72rem; white-space:nowrap;'>{upd_str}</td>"
            f"<td style='white-space:nowrap;'>{close_btn}{edit_btn}{del_btn}</td>"
            f"</tr>"
        )

    table = (
        "<style>"
        "table.trades-tbl{width:100%;border-collapse:collapse;font-size:0.83rem;}"
        "table.trades-tbl th{background:#161b22;color:#4a5568;padding:8px 10px;"
        "border-bottom:1px solid #21262d;text-align:left;font-size:0.68rem;"
        "font-weight:700;letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;}"
        "table.trades-tbl td{background:#0d1117;color:#e2e8f4;padding:9px 10px;"
        "border-bottom:1px solid #161b22;vertical-align:middle;}"
        "table.trades-tbl tr:hover td{background:#13181f;}"
        "</style>"
        "<table class='trades-tbl'><thead><tr>"
        "<th>SYMBOL</th><th>SIDE</th><th>STATUS</th>"
        "<th>ENTRY / QTY</th><th>CURRENT</th><th>P&amp;L</th>"
        "<th>TARGET / SL</th><th>R:R</th><th>STRATEGY</th>"
        "<th>NOTES</th><th>UPDATED</th><th>ACTIONS</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return summary + table


# ── Gradio action handlers ────────────────────────────────────────────────────

def add_trade_action(symbol: str, strategy: str, side: str, status: str,
                     entry_price: str, quantity: str, target: str, stop_loss: str,
                     notes: str, tags: str,
                     status_filter: str, sort_by: str) -> tuple[str, str]:
    msg = add_trade(symbol, strategy, side, status, entry_price, quantity, target, stop_loss, notes, tags)
    return msg, build_trades_html(status_filter, sort_by)


def close_trade_action(trade_id: str, exit_price: str,
                       status_filter: str, sort_by: str) -> tuple[str, str]:
    msg = close_trade(trade_id, exit_price)
    return msg, build_trades_html(status_filter, sort_by)


def delete_trade_action(trade_id: str, status_filter: str, sort_by: str) -> tuple[str, str]:
    msg = delete_trade(trade_id)
    return msg, build_trades_html(status_filter, sort_by)


def refresh_prices_action(status_filter: str, sort_by: str) -> tuple[str, str]:
    msg = refresh_open_prices()
    return msg, build_trades_html(status_filter, sort_by)


def refresh_trades_table_action(status_filter: str, sort_by: str) -> str:
    return build_trades_html(status_filter, sort_by)


def save_edit_action(trade_id: str, strategy: str, notes: str, tags: str,
                     status_filter: str, sort_by: str) -> tuple[str, str]:
    msg = update_trade_fields(trade_id, strategy, notes, tags)
    return msg, build_trades_html(status_filter, sort_by)
