from datetime import datetime
from pymongo import MongoClient, DESCENDING
from .config import MONGO_URI

# ── Registry of algobots (add more here later) ────────────────────────────────
BOTS = {
    "IntradayMomentum": {
        "label":      "Intraday Momentum",
        "db":         "IntradayMomentum",
        "collection": "trades",
        "pnl_field":  "gross_pnl",
        "date_field": "entry_date",      # string "YYYY-MM-DD"
        "ticker_field": "ticker",
        "status_field": "status",
        "closed_value": "closed",
    },
}


def _col(bot_key: str):
    cfg    = BOTS[bot_key]
    client = MongoClient(MONGO_URI)
    return client[cfg["db"]][cfg["collection"]], cfg


# ── Summary: one row per trading day ─────────────────────────────────────────

def get_summary(bot_key: str) -> dict:
    if bot_key not in BOTS:
        return {"ok": False, "error": f"Unknown bot: {bot_key}"}

    col, cfg = _col(bot_key)
    pnl_f    = cfg["pnl_field"]
    date_f   = cfg["date_field"]
    status_f = cfg["status_field"]
    closed_v = cfg["closed_value"]

    all_trades = list(col.find())
    if not all_trades:
        return {"ok": True, "days": [], "totals": _zero_totals()}

    # Group by date
    by_date: dict = {}
    for t in all_trades:
        d = t.get(date_f, "Unknown")
        if isinstance(d, datetime):
            d = d.strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append(t)

    def _capital(trade: dict) -> float:
        """Capital deployed in one trade = entry_price × shares."""
        ep  = float(trade.get("entry_price", 0) or 0)
        qty = float(trade.get("shares", 0) or trade.get("quantity", 0) or 0)
        return round(ep * qty, 2)

    days = []
    for date_str in sorted(by_date.keys(), reverse=True):
        trades  = by_date[date_str]
        closed  = [t for t in trades if t.get(status_f, "").lower() == closed_v]
        open_t  = [t for t in trades if t.get(status_f, "").lower() != closed_v]

        pnls    = [float(t.get(pnl_f, 0) or 0) for t in closed]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p < 0]

        total_pnl  = round(sum(pnls), 2)
        win_rate   = round(len(wins) / len(pnls) * 100, 1) if pnls else 0
        max_loss   = round(min(pnls), 2) if losses else 0
        best_trade = round(max(pnls), 2) if wins else 0
        avg_pnl    = round(sum(pnls) / len(pnls), 2) if pnls else 0

        # Capital metrics
        capital_locked   = round(sum(_capital(t) for t in open_t), 2)   # still open
        capital_deployed = round(sum(_capital(t) for t in trades), 2)   # all (open+closed) that day
        avg_trade_size   = round(capital_deployed / len(trades), 2) if trades else 0

        tickers = sorted({t.get(cfg["ticker_field"], "") for t in trades})

        days.append({
            "date":             date_str,
            "total":            len(trades),
            "closed":           len(closed),
            "open":             len(open_t),
            "win_rate":         win_rate,
            "total_pnl":        total_pnl,
            "max_loss":         max_loss,
            "best_trade":       best_trade,
            "avg_pnl":          avg_pnl,
            "capital_locked":   capital_locked,
            "capital_deployed": capital_deployed,
            "avg_trade_size":   avg_trade_size,
            "tickers":          tickers[:6],
        })

    # Overall totals
    all_closed  = [t for t in all_trades if t.get(status_f, "").lower() == closed_v]
    all_open    = [t for t in all_trades if t.get(status_f, "").lower() != closed_v]
    all_pnls    = [float(t.get(pnl_f, 0) or 0) for t in all_closed]
    total_locked    = round(sum(_capital(t) for t in all_open), 2)
    total_deployed  = round(sum(_capital(t) for t in all_trades), 2)
    totals = {
        "total_trades":     len(all_trades),
        "closed_trades":    len(all_closed),
        "open_trades":      len(all_open),
        "total_pnl":        round(sum(all_pnls), 2),
        "win_rate":         round(len([p for p in all_pnls if p>0]) / len(all_pnls) * 100, 1) if all_pnls else 0,
        "max_loss":         round(min(all_pnls), 2) if all_pnls else 0,
        "best_trade":       round(max(all_pnls), 2) if all_pnls else 0,
        "avg_pnl":          round(sum(all_pnls) / len(all_pnls), 2) if all_pnls else 0,
        "trading_days":     len(days),
        "capital_locked":   total_locked,
        "capital_deployed": total_deployed,
        "avg_trade_size":   round(total_deployed / len(all_trades), 2) if all_trades else 0,
    }

    return {"ok": True, "days": days, "totals": totals}


# ── Detail: all trades for a specific date ────────────────────────────────────

def get_day_trades(bot_key: str, date_str: str) -> dict:
    if bot_key not in BOTS:
        return {"ok": False, "error": f"Unknown bot: {bot_key}"}

    col, cfg  = _col(bot_key)
    date_f    = cfg["date_field"]
    pnl_f     = cfg["pnl_field"]

    trades = list(col.find({date_f: date_str}).sort("entry_time", 1))
    out = []
    for t in trades:
        t["id"] = str(t.pop("_id"))
        for fld in ("entry_timestamp", "exit_timestamp", "updated_at"):
            if isinstance(t.get(fld), datetime):
                t[fld] = t[fld].isoformat()
        # Normalise pnl
        t["pnl"] = round(float(t.get(pnl_f, 0) or 0), 2)
        out.append(t)

    return {"ok": True, "trades": out, "date": date_str}


def _zero_totals() -> dict:
    return {"total_trades": 0, "closed_trades": 0, "open_trades": 0,
            "total_pnl": 0, "win_rate": 0, "max_loss": 0, "best_trade": 0,
            "avg_pnl": 0, "trading_days": 0,
            "capital_locked": 0, "capital_deployed": 0, "avg_trade_size": 0}


# ── IBKR Pending Orders ───────────────────────────────────────────────────────

def get_pending_orders() -> dict:
    """Fetch all pending orders from IBKR with full details."""
    try:
        from .db import ib

        if not ib.isConnected():
            return {
                "ok": True,
                "orders": [],
                "status": "IBKR not connected",
                "total": 0,
            }

        # Fetch all open trades (orders)
        open_orders = []
        for trade in ib.openTrades():
            try:
                order = trade.order
                order_status = trade.orderStatus

                # Basic order details
                contract = order.contract
                symbol = contract.symbol

                qty = float(order.totalQuantity)
                filled = float(order.filledQuantity)
                remaining = qty - filled

                # Price info
                limit_price = float(order.lmtPrice) if order.lmtPrice else None
                aux_price = float(order.auxPrice) if order.auxPrice else None

                # Status
                status = order_status.status
                avg_fill_price = float(order_status.avgFillPrice) if order_status.avgFillPrice else None

                open_orders.append({
                    "order_id": order.orderId,
                    "symbol": symbol,
                    "side": order.action,  # BUY or SELL
                    "quantity": int(qty),
                    "filled": int(filled),
                    "remaining": int(remaining),
                    "limit_price": round(limit_price, 2) if limit_price else None,
                    "aux_price": round(aux_price, 2) if aux_price else None,
                    "avg_fill_price": round(avg_fill_price, 2) if avg_fill_price else None,
                    "status": status,
                    "order_type": order.orderType,
                    "time_in_force": order.tif,
                    "created_time": order.createdTime if hasattr(order, 'createdTime') else None,
                    "percent_filled": round(filled / qty * 100, 1) if qty > 0 else 0,
                })
            except Exception as e:
                print(f"Error processing order {trade.order.orderId}: {e}")
                continue

        # Sort by symbol
        open_orders.sort(key=lambda x: x["symbol"])

        return {
            "ok": True,
            "orders": open_orders,
            "total": len(open_orders),
            "status": "connected",
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "orders": [],
            "total": 0,
        }
