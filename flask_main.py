from dotenv import load_dotenv
load_dotenv()

import re
import threading
import concurrent.futures
from datetime import datetime, timedelta
from bson import ObjectId
import numpy as np
from flask import Flask, jsonify, request, render_template
from pymongo import MongoClient, DESCENDING as _DESC

from alphaterminal.db import alerts_collection, analysis_collection, trades_collection
from alphaterminal.config import MONGO_URI as _MONGO_URI

# ── Strategy Lab saved results collection ────────────────────────────────────
_sl_col = MongoClient(_MONGO_URI)["market_watchlist"]["strategy_lab_saved"]
_sl_col.create_index([("saved_at", _DESC)])
from alphaterminal.trades import (
    _calc_derived, add_trade, close_trade, delete_trade,
    update_trade_fields, get_trades, refresh_open_prices,
)
from alphaterminal.notes import add_note, delete_note, get_all_notes
from alphaterminal.new_finds import (
    add_find, update_status as update_find_status,
    delete_find, get_finds, get_weeks, analyze_find,
    get_weekly_recommendations, _current_week,
    SOURCES, CATEGORIES, SENTIMENTS,
)
from alphaterminal.algobot_report import get_pending_orders
from alphaterminal.crypto import get_coins_enriched, add_coin, remove_coin
from alphaterminal.alerts import fetch_and_store_alerts
from alphaterminal.analysis import (
    fetch_stock_data, get_or_fetch_analysis,
    _parse_recommendation, _parse_confidence, _REC_NORM,
    tradingview_embed_html, render_analysis_html,
)
from alphaterminal.watchlist import (
    get_watchlist_symbols, add_watchlist_symbol, remove_watchlist_symbol,
)
from alphaterminal.data import fetch_market_data
from alphaterminal.data import fetch_fear_greed_index
from alphaterminal.order import place_order
from alphaterminal.utils import normalize_symbol

app = Flask(__name__, template_folder="templates")

# Teach Flask's jsonify to handle numpy scalar types (float64, int64, etc.)
# returned by pandas .mean(), .std(), round(), etc.
try:
    from flask.json.provider import DefaultJSONProvider
    class _NumpyProvider(DefaultJSONProvider):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    app.json_provider_class = _NumpyProvider
    app.json = _NumpyProvider(app)
except ImportError:
    from flask.json import JSONEncoder  # type: ignore
    class _NumpyEncoder(JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    app.json_encoder = _NumpyEncoder  # type: ignore

# ── In-memory analyst cache (5-minute TTL) ────────────────────────────────────
_analyst_cache: dict       = {}
_analyst_cache_ts: dict    = {}
_ANALYST_TTL               = 300


def _get_analyst(symbol: str) -> dict:
    now = datetime.now().timestamp()
    if symbol in _analyst_cache and (now - _analyst_cache_ts.get(symbol, 0)) < _ANALYST_TTL:
        return _analyst_cache[symbol]

    sd = fetch_stock_data(symbol)

    # Recommendation — prefer cached GPT analysis, fall back to yfinance
    rec  = "—"
    conf = None
    doc  = analysis_collection.find_one({"ticker": symbol})
    if doc:
        text     = doc.get("analysis", "")
        rec      = _parse_recommendation(text, sd)
        conf_str = _parse_confidence(text)
        conf     = int(conf_str) if conf_str.isdigit() else None
    else:
        m = re.search(r"Consensus:\s*([^\n]+)", sd.get("analyst_summary", ""), re.IGNORECASE)
        if m:
            raw = m.group(1).strip().lower().replace("_", "").replace(" ", "")
            for key, val in _REC_NORM.items():
                if key.replace("_", "").replace(" ", "") == raw:
                    rec = val
                    break

    # Price / change
    price      = sd.get("current_price")
    change_str = sd.get("daily_change") or ""
    change_pct = None
    if change_str:
        try:
            change_pct = float(change_str.replace("%", "").replace("+", ""))
        except ValueError:
            pass

    # Analyst target
    target = None
    m2 = re.search(r"Mean Target:\s*\$?([\d.]+)", sd.get("analyst_summary", ""))
    if m2:
        target = float(m2.group(1))

    target_upside = round((target - price) / price * 100, 1) if (target and price) else None

    result = {
        "price":            price,
        "change_pct":       change_pct,
        "target":           target,
        "target_upside_pct": target_upside,
        "recommendation":   rec,
        "confidence":       conf,
    }
    _analyst_cache[symbol]    = result
    _analyst_cache_ts[symbol] = now
    return result


def _safe_analyst(symbol: str) -> dict:
    try:
        return _get_analyst(symbol)
    except Exception:
        return {"price": None, "change_pct": None, "target": None,
                "target_upside_pct": None, "recommendation": "—", "confidence": None}


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/alerts")
def alerts_panel():
    return render_template("alerts_panel.html")

@app.get("/crypto")
def crypto_panel():
    return render_template("crypto_panel.html")


# ── Data API ──────────────────────────────────────────────────────────────────

@app.get("/api/fear_greed")
def api_fear_greed():
    raw = str(fetch_fear_greed_index())
    # Typical format: "48 — Neutral" or "Extreme Fear (25)"
    m = re.search(r"(\d+)", raw)
    value = m.group(1) if m else "—"
    parts = re.split(r"[—\-–]", raw, maxsplit=1)
    label = parts[1].strip() if len(parts) > 1 else raw.strip()
    return jsonify({"value": value, "label": label, "full": raw})


@app.get("/api/analyst/<ticker>")
def api_analyst(ticker: str):
    sym = normalize_symbol(ticker.upper())
    return jsonify(_safe_analyst(sym))


@app.get("/api/alerts")
def api_alerts():
    now  = datetime.now()
    docs = list(alerts_collection.find({"expires_at": {"$gt": now}}))

    # Deduplicate: sort oldest-first, keep earliest per (group, symbol)
    docs.sort(key=lambda d: d.get("received_at") or datetime.min)
    seen_keys: set = set()
    deduped: list  = []
    for doc in docs:
        key = (doc.get("watchlist", "Unknown"), doc["symbol"])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(doc)
    docs = deduped

    # Unique symbols for concurrent analyst fetch
    symbols = list({d["symbol"] for d in docs})
    analyst_data: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        fs = {ex.submit(_safe_analyst, s): s for s in symbols}
        for f in concurrent.futures.as_completed(fs):
            analyst_data[fs[f]] = f.result()

    def _fmt(dt) -> str | None:
        return dt.strftime("%m/%d %H:%M") if dt else None

    # Group by watchlist (= group_name)
    groups: dict = {}
    for doc in docs:
        gname = doc.get("watchlist") or doc.get("group_name", "Unknown")
        groups.setdefault(gname, []).append({
            "id":              str(doc["_id"]),
            "symbol":          doc["symbol"],
            "group":           gname,
            "exp_date":        doc["expires_at"].strftime("%m/%d") if doc.get("expires_at") else None,
            "status":          doc.get("status", "new"),
            "alert_generated": _fmt(doc.get("received_at")),
            "traded_at":       _fmt(doc.get("traded_at")),
            "analyst":         analyst_data.get(doc["symbol"], {}),
        })

    _STATUS_ORDER = {"new": 0, "queued": 1, "traded": 2, "skipped": 3}
    result = []
    for gname, alerts in sorted(groups.items()):
        alerts.sort(key=lambda a: (_STATUS_ORDER.get(a["status"], 9), a["symbol"]))
        new_count = sum(1 for a in alerts if a["status"] == "new")
        result.append({
            "group_name":  gname,
            "total_count": len(alerts),
            "new_count":   new_count,
            "alerts":      alerts,
        })
    result.sort(key=lambda g: -g["new_count"])
    return jsonify({"groups": result})


# ── Analyst batch fetch ───────────────────────────────────────────────────────

_batch_lock    = threading.Lock()
_batch_state: dict = {"running": False, "pending": [], "done": [], "errors": []}


def _active_alert_symbols() -> list[str]:
    return list({d["symbol"] for d in alerts_collection.find({"expires_at": {"$gt": datetime.now()}})})


def _missing_symbols() -> list[str]:
    """Symbols in active alerts that have no cached GPT analysis (within 5-day TTL)."""
    limit = datetime.now() - timedelta(days=5)
    all_syms = _active_alert_symbols()
    return [s for s in all_syms
            if not analysis_collection.find_one({"ticker": s, "fetched_at": {"$gt": limit}})]


def _run_batch_analyst(symbols: list[str]) -> None:
    with _batch_lock:
        _batch_state.update(running=True, pending=list(symbols), done=[], errors=[])

    for sym in symbols:
        try:
            get_or_fetch_analysis(sym)
            # Invalidate in-memory analyst cache so next /api/alerts gets fresh data
            _analyst_cache.pop(sym, None)
            _analyst_cache_ts.pop(sym, None)
            with _batch_lock:
                _batch_state["done"].append(sym)
                if sym in _batch_state["pending"]:
                    _batch_state["pending"].remove(sym)
        except Exception as exc:
            with _batch_lock:
                _batch_state["errors"].append({"symbol": sym, "error": str(exc)})
                if sym in _batch_state["pending"]:
                    _batch_state["pending"].remove(sym)

    with _batch_lock:
        _batch_state["running"] = False


@app.get("/api/alerts/analyst-status")
def api_analyst_status():
    missing = _missing_symbols()
    with _batch_lock:
        state = dict(_batch_state)
    return jsonify({
        "missing_count": len(missing),
        "missing":       missing,
        "running":       state["running"],
        "pending":       state["pending"],
        "done":          state["done"],
        "errors":        state["errors"],
    })


@app.post("/api/alerts/fetch-analyst")
def api_fetch_analyst():
    with _batch_lock:
        if _batch_state["running"]:
            return jsonify({"ok": False, "error": "Already running — check status."}), 409

    body    = request.get_json() or {}
    symbols = body.get("symbols") or _missing_symbols()

    if not symbols:
        return jsonify({"ok": True, "message": "No missing analyst data found.", "count": 0})

    threading.Thread(target=_run_batch_analyst, args=(symbols,), daemon=True).start()
    return jsonify({"ok": True, "count": len(symbols), "symbols": symbols,
                    "message": f"Fetching analyst data for {len(symbols)} symbols in background…"})


# ── Action helpers ────────────────────────────────────────────────────────────

def _trade_symbol(symbol: str, group: str) -> dict:
    """Place order, mark alert traded, and auto-log to Trade Monitor."""
    result  = place_order(symbol)
    now     = datetime.now()

    alerts_collection.update_many(
        {"symbol": symbol, "watchlist": group},
        {"$set": {"status": "traded", "traded_at": now}},
    )

    # Auto-log to Trade Monitor (skip if an open alert-sourced trade already exists)
    existing = trades_collection.find_one({"symbol": symbol, "status": "OPEN", "source": "alert"})
    if not existing:
        ad    = _safe_analyst(symbol)
        price = ad.get("price") or 0
        doc   = {
            "symbol":             symbol,
            "strategy_name":      f"Alert — {group}",
            "side":               "LONG",
            "status":             "OPEN",
            "entry_price":        price,
            "entry_date":         now,
            "quantity":           1,
            "target_price":       ad.get("target"),
            "stop_loss":          None,
            "realized_pnl":       None,
            "exit_price":         None,
            "exit_date":          None,
            "notes":              f"Auto-logged from alert group: {group}",
            "tags":               ["alert", "auto"],
            "source":             "alert",
            "created_at":         now,
            "highest_price_seen": None,
            "lowest_price_seen":  None,
        }
        doc.update(_calc_derived(doc, price) if price else {"cost": 0, "unrealized_pnl": 0,
            "pnl_pct": 0, "highest_price_seen": None, "lowest_price_seen": None,
            "max_profit_pct": None, "max_loss_pct": None, "risk_reward": None,
            "current_price": 0, "updated_at": now})
        trades_collection.insert_one(doc)

    return result


def _set_status(symbol: str, group: str, status: str) -> None:
    alerts_collection.update_many(
        {"symbol": symbol, "watchlist": group},
        {"$set": {"status": status, "updated_at": datetime.now()}},
    )


# ── Action endpoints ──────────────────────────────────────────────────────────

@app.post("/api/alerts/trade")
def api_trade():
    data   = request.get_json() or {}
    symbol = normalize_symbol(data.get("symbol", ""))
    group  = data.get("group", "")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    order = _trade_symbol(symbol, group)
    return jsonify({"ok": True, "order": order})


@app.post("/api/alerts/skip")
def api_skip():
    data   = request.get_json() or {}
    symbol = normalize_symbol(data.get("symbol", ""))
    group  = data.get("group", "")
    _set_status(symbol, group, "skipped")
    return jsonify({"ok": True})


@app.post("/api/alerts/readd")
def api_readd():
    data   = request.get_json() or {}
    symbol = normalize_symbol(data.get("symbol", ""))
    group  = data.get("group", "")
    _set_status(symbol, group, "new")
    return jsonify({"ok": True})


@app.post("/api/alerts/trade-group")
def api_trade_group():
    data       = request.get_json() or {}
    group_name = data.get("group_name", "")
    targets    = list(alerts_collection.find({
        "watchlist":  group_name,
        "status":     "new",
        "expires_at": {"$gt": datetime.now()},
    }))
    results = []
    for doc in targets:
        sym = doc["symbol"]
        try:
            results.append({"symbol": sym, "ok": True, "order": _trade_symbol(sym, group_name)})
        except Exception as exc:
            results.append({"symbol": sym, "ok": False, "error": str(exc)})
    return jsonify({"ok": True, "results": results, "count": len(results)})


@app.post("/api/alerts/trade-all")
def api_trade_all():
    targets = list(alerts_collection.find({
        "status":     "new",
        "expires_at": {"$gt": datetime.now()},
    }))
    results = []
    for doc in targets:
        sym   = doc["symbol"]
        group = doc.get("watchlist", "")
        try:
            results.append({"symbol": sym, "group": group, "ok": True,
                            "order": _trade_symbol(sym, group)})
        except Exception as exc:
            results.append({"symbol": sym, "group": group, "ok": False, "error": str(exc)})
    return jsonify({"ok": True, "results": results, "count": len(results)})


@app.post("/api/alerts/fetch")
def api_fetch():
    """Fetch from Fastmail; if auto_trade=true, queue+trade every new symbol."""
    body       = request.get_json() or {}
    auto_trade = body.get("auto_trade", False)
    msg        = fetch_and_store_alerts()
    traded     = []

    if auto_trade:
        new_docs = list(alerts_collection.find({
            "status":     "new",
            "expires_at": {"$gt": datetime.now()},
        }))
        for doc in new_docs:
            sym   = doc["symbol"]
            group = doc.get("watchlist", "")
            _set_status(sym, group, "queued")
            try:
                _trade_symbol(sym, group)
                traded.append(sym)
            except Exception:
                pass

    return jsonify({"ok": True, "message": msg, "auto_traded": traded})


# ═══════════════════════════════════════════════════════════════════════════════
# Watchlist & Dip-Buy
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/watchlist")
def watchlist_panel():
    return render_template("watchlist_panel.html")


@app.get("/api/watchlist/symbols")
def api_watchlist_symbols():
    return jsonify({"symbols": get_watchlist_symbols()})


@app.get("/api/watchlist/data")
def api_watchlist_data():
    symbols = get_watchlist_symbols()
    if not symbols:
        return jsonify({"rows": []})
    market_data = fetch_market_data(symbols, normalize_symbol)
    rows = []
    for sym in symbols:
        d = market_data.get(sym, {})
        if d.get("error"):
            rows.append({"symbol": sym, "error": True})
        else:
            rows.append({
                "symbol":              sym,
                "price":               d.get("price"),
                "change_pct":          d.get("change_pct"),
                "change_7d":           d.get("change_7d"),
                "change_30d":          d.get("change_30d"),
                "rsi":                 d.get("rsi"),
                "below_volume_profile": d.get("below_volume_profile"),
                "support":             d.get("support"),
                "resistance":          d.get("resistance"),
                "tradingview_url":     d.get("tradingview_url"),
                "error":               False,
            })
    return jsonify({"rows": rows})


@app.get("/api/watchlist/analyst-data")
def api_watchlist_analyst_data():
    """Return cached analyst data for all watchlist symbols (reads MongoDB, no live fetch)."""
    from alphaterminal.analysis import _parse_recommendation, _parse_confidence, _news_highlight
    symbols = get_watchlist_symbols()
    result  = {}
    for sym in symbols:
        doc = analysis_collection.find_one({"ticker": sym})
        if not doc:
            result[sym] = {"stale": True}
            continue
        text = doc.get("analysis", "")
        sd   = doc.get("stock_data", {})
        rec  = _parse_recommendation(text, sd)
        conf_str = _parse_confidence(text)
        conf = int(conf_str) if conf_str.isdigit() else None

        target = None
        m2 = re.search(r"Mean Target:\s*\$?([\d.]+)", sd.get("analyst_summary", ""))
        if m2:
            target = float(m2.group(1))

        price  = sd.get("current_price")
        upside = round((target - price) / price * 100, 1) if (target and price) else None

        fetched_at = doc.get("fetched_at")
        age_h = round((datetime.now() - fetched_at).total_seconds() / 3600, 1) if fetched_at else None

        result[sym] = {
            "recommendation": rec,
            "confidence":     conf,
            "target":         target,
            "upside_pct":     upside,
            "news":           _news_highlight(text, sd),
            "fetched_at":     fetched_at.isoformat() if fetched_at else None,
            "age_hours":      age_h,
            "stale":          (age_h is None) or (age_h > 120),
        }
    return jsonify(result)


@app.post("/api/watchlist/refresh-analyst")
def api_watchlist_refresh_analyst():
    """Kick off analyst refresh for one symbol or the entire watchlist."""
    b      = request.get_json() or {}
    symbol = b.get("symbol", "").strip().upper()
    if symbol:
        symbols = [normalize_symbol(symbol)]
    else:
        symbols = get_watchlist_symbols()

    with _batch_lock:
        if _batch_state["running"]:
            return jsonify({"ok": False, "error": "Already running — check /api/alerts/analyst-status"}), 409

    if not symbols:
        return jsonify({"ok": False, "error": "No symbols in watchlist"}), 400

    threading.Thread(target=_run_batch_analyst, args=(symbols,), daemon=True).start()
    return jsonify({"ok": True, "count": len(symbols), "symbols": symbols,
                    "message": f"Refreshing analyst data for {len(symbols)} symbol(s) in background…"})


@app.post("/api/watchlist")
def api_watchlist_add():
    b   = request.get_json() or {}
    sym = normalize_symbol(b.get("symbol", "").strip().upper())
    if not sym:
        return jsonify({"error": "symbol required"}), 400
    msg = add_watchlist_symbol(sym)
    return jsonify({"ok": True, "message": msg})


@app.delete("/api/watchlist/<symbol>")
def api_watchlist_remove(symbol: str):
    remove_watchlist_symbol(symbol.upper())
    return jsonify({"ok": True})


@app.get("/api/analyze/<ticker>")
def api_analyze(ticker: str):
    ticker = normalize_symbol(ticker.strip().upper())
    try:
        analysis_text, stock_data = get_or_fetch_analysis(ticker)
        chart_html    = tradingview_embed_html(ticker)
        analysis_html = render_analysis_html(ticker, analysis_text, stock_data)
        return jsonify({"chart_html": chart_html, "analysis_html": analysis_html})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# FIRE & Retirement Planner
# ═══════════════════════════════════════════════════════════════════════════════
from alphaterminal.fire import (
    get_portfolios, create_portfolio, delete_portfolio,
    add_holding, remove_holding, update_holding, get_holdings_enriched,
    get_suggestions,
)


@app.get("/fire")
def fire_panel():
    return render_template("fire_panel.html")


@app.get("/api/fire/portfolios")
def api_fire_portfolios():
    return jsonify(get_portfolios())


@app.post("/api/fire/portfolios")
def api_fire_create_portfolio():
    b = request.get_json() or {}
    pid = create_portfolio(b.get("name","New Portfolio"), b.get("type","CUSTOM"),
                           b.get("description",""), b.get("color","#7a8499"))
    return jsonify({"ok": True, "id": pid})


@app.delete("/api/fire/portfolios/<pid>")
def api_fire_delete_portfolio(pid: str):
    delete_portfolio(pid)
    return jsonify({"ok": True})


@app.get("/api/fire/<pid>/holdings")
def api_fire_holdings(pid: str):
    from alphaterminal.db import fire_portfolios_collection
    from bson import ObjectId
    try:
        port = fire_portfolios_collection.find_one({"_id": ObjectId(pid)})
        ptype = port.get("type", "") if port else ""
    except Exception:
        ptype = ""
    return jsonify(get_holdings_enriched(pid, ptype))


@app.post("/api/fire/<pid>/holdings")
def api_fire_add_holding(pid: str):
    b = request.get_json() or {}
    sym = normalize_symbol(b.get("symbol", ""))
    if not sym:
        return jsonify({"error": "symbol required"}), 400
    hid = add_holding(
        pid, sym,
        status           = b.get("status", "watching"),
        target_alloc_pct = b.get("target_alloc_pct"),
        dca_target_price = b.get("dca_target_price"),
        shares_owned     = b.get("shares_owned"),
        avg_cost         = b.get("avg_cost"),
        notes            = b.get("notes", ""),
        tags             = b.get("tags", []),
    )
    return jsonify({"ok": True, "id": hid})


@app.delete("/api/fire/<pid>/holdings/<symbol>")
def api_fire_remove_holding(pid: str, symbol: str):
    remove_holding(pid, symbol)
    return jsonify({"ok": True})


@app.patch("/api/fire/<pid>/holdings/<symbol>")
def api_fire_update_holding(pid: str, symbol: str):
    update_holding(pid, symbol, request.get_json() or {})
    return jsonify({"ok": True})


@app.get("/api/fire/suggestions")
def api_fire_suggestions():
    return jsonify(get_suggestions())


# ═══════════════════════════════════════════════════════════════════════════════
# Crypto Monitor
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/crypto")
def api_crypto():
    try:
        coins = get_coins_enriched()
        return jsonify({"ok": True, "coins": coins})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/crypto")
def api_crypto_add():
    b = request.get_json() or {}
    symbol = b.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    cid = add_coin(symbol, name=b.get("name", ""), color=b.get("color", "#7a8499"))
    return jsonify({"ok": True, "id": cid})


@app.delete("/api/crypto/<symbol>")
def api_crypto_remove(symbol: str):
    remove_coin(symbol.upper())
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Lab
# ═══════════════════════════════════════════════════════════════════════════════

from alphaterminal.strategy_lab import (
    start_lab_job, get_job,
    get_strategy_library, suggest_strategies, web_search_strategies,
)


@app.get("/strategy-lab")
def strategy_lab_page():
    resp = app.make_response(render_template("strategy_lab.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.post("/api/strategy-lab/run")
def api_strategy_lab_run():
    body = request.get_json() or {}
    mode = body.get("mode", "validate")
    if mode == "validate" and not body.get("code", "").strip():
        return jsonify({"error": "code required for validate mode"}), 400
    if mode == "generate" and not body.get("idea", "").strip():
        return jsonify({"error": "idea required for generate mode"}), 400
    job_id = start_lab_job(body)
    return jsonify({"job_id": job_id})


@app.get("/api/strategy-lab/status/<job_id>")
def api_strategy_lab_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status":   job.get("status"),
        "step":     job.get("step"),
        "progress": job.get("progress", 0),
        "log":      job.get("log", []),
        "error":    job.get("error"),
        "tokens":   job.get("tokens", {}),
    })


@app.get("/api/strategy-lab/library")
def api_strategy_library():
    return jsonify(get_strategy_library())


@app.get("/api/strategy-lab/library-prompt")
def api_strategy_library_prompt():
    from alphaterminal.strategy_lab import STRATEGY_LIBRARY
    sid = request.args.get("id", "")
    entry = next((s for s in STRATEGY_LIBRARY if s["id"] == sid), None)
    if not entry:
        return jsonify({"error": "not found"}), 404
    return jsonify({"idea_prompt": entry.get("idea_prompt", ""), "name": entry["name"]})


@app.post("/api/strategy-lab/web-search")
def api_strategy_web_search():
    body  = request.get_json() or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    result = web_search_strategies(query)
    return jsonify(result)


@app.post("/api/strategy-lab/suggest")
def api_strategy_suggest():
    body = request.get_json() or {}
    query = body.get("query", "").strip()
    market_context = body.get("market_context", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    result = suggest_strategies(query, market_context)
    return jsonify(result)


@app.get("/api/strategy-lab/result/<job_id>")
def api_strategy_lab_result(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "not ready", "status": job.get("status")}), 202
    return jsonify(job.get("result") or {})


# ── Strategy Lab — saved results ──────────────────────────────────────────────

@app.post("/api/strategy-lab/save")
def api_strategy_lab_save():
    body   = request.get_json() or {}
    result = body.get("result")
    if not result:
        return jsonify({"error": "result payload required"}), 400
    name  = (body.get("name") or "").strip() or "Untitled"
    notes = (body.get("notes") or "").strip()

    # Pull lightweight summary fields so the list view is fast
    consensus = result.get("consensus") or {}
    v1m       = (result.get("v1") or {}).get("metrics") or {}
    parsed    = result.get("parsed") or {}

    doc = {
        "name":       name,
        "notes":      notes,
        "saved_at":   datetime.utcnow(),
        "mode":       result.get("mode", "validate"),
        "verdict":    consensus.get("verdict", ""),
        "confidence": consensus.get("confidence_score"),
        "sharpe":     v1m.get("sharpe"),
        "cagr":       v1m.get("cagr_pct"),
        "max_dd":     v1m.get("max_drawdown_pct"),
        "strategy_name": parsed.get("strategy_name", name),
        "symbols":    (result.get("config") or {}).get("symbols", []),
        "best_version": result.get("best_version", 1),
        "result":     result,   # full payload
    }
    inserted = _sl_col.insert_one(doc)
    return jsonify({"ok": True, "id": str(inserted.inserted_id)})


@app.get("/api/strategy-lab/saved")
def api_strategy_lab_saved_list():
    docs = _sl_col.find(
        {}, {"result": 0}          # exclude heavy result field from list
    ).sort("saved_at", _DESC).limit(100)
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        if isinstance(d.get("saved_at"), datetime):
            d["saved_at"] = d["saved_at"].isoformat()
        out.append(d)
    return jsonify(out)


@app.get("/api/strategy-lab/saved/<save_id>")
def api_strategy_lab_saved_get(save_id: str):
    try:
        oid = ObjectId(save_id)
    except Exception:
        return jsonify({"error": "invalid id"}), 400
    doc = _sl_col.find_one({"_id": oid})
    if not doc:
        return jsonify({"error": "not found"}), 404
    doc["id"] = str(doc.pop("_id"))
    if isinstance(doc.get("saved_at"), datetime):
        doc["saved_at"] = doc["saved_at"].isoformat()
    return jsonify(doc)


@app.delete("/api/strategy-lab/saved/<save_id>")
def api_strategy_lab_saved_delete(save_id: str):
    try:
        oid = ObjectId(save_id)
    except Exception:
        return jsonify({"error": "invalid id"}), 400
    res = _sl_col.delete_one({"_id": oid})
    if res.deleted_count == 0:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════════
# Notes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/notes")
def notes_panel():
    return render_template("notes_panel.html")


@app.get("/api/notes")
def api_notes_list():
    notes = get_all_notes()
    out = []
    for n in notes:
        n["id"] = str(n.pop("_id"))
        if isinstance(n.get("created_at"), datetime):
            n["created_at"] = n["created_at"].isoformat()
        out.append(n)
    return jsonify(out)


@app.post("/api/notes")
def api_notes_add():
    b      = request.get_json() or {}
    ticker = b.get("ticker", "").strip().upper()
    text   = b.get("note", "").strip()
    if not text:
        return jsonify({"error": "note text required"}), 400
    msg = add_note(ticker, text)
    return jsonify({"ok": True, "message": msg})


@app.delete("/api/notes/<note_id>")
def api_notes_delete(note_id: str):
    msg = delete_note(note_id)
    return jsonify({"ok": True, "message": msg})


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Monitor
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/trades")
def trades_panel():
    return render_template("trades_panel.html")


@app.get("/api/trades")
def api_trades_list():
    status_filter = request.args.get("status", "ALL").upper()
    sort_by       = request.args.get("sort", "entry_date")
    trades        = get_trades(status_filter, sort_by)
    out = []
    for t in trades:
        t["id"] = str(t.pop("_id"))
        for fld in ("entry_date", "exit_date", "created_at", "updated_at"):
            if isinstance(t.get(fld), datetime):
                t[fld] = t[fld].isoformat()
        out.append(t)
    return jsonify(out)


@app.post("/api/trades")
def api_trades_add():
    b   = request.get_json() or {}
    msg = add_trade(
        b.get("symbol", ""),
        b.get("strategy_name", ""),
        b.get("side", "LONG"),
        b.get("status", "OPEN"),
        str(b.get("entry_price", "")),
        str(b.get("quantity", "0")),
        str(b.get("target_price", "")),
        str(b.get("stop_loss", "")),
        b.get("notes", ""),
        b.get("tags", ""),
    )
    ok = not msg.startswith(("Symbol", "Valid"))
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)


@app.post("/api/trades/<trade_id>/close")
def api_trades_close(trade_id: str):
    b          = request.get_json() or {}
    exit_price = str(b.get("exit_price", "")).strip()
    msg        = close_trade(trade_id, exit_price)
    ok         = not any(msg.startswith(p) for p in ("Could not", "Invalid", "Trade not"))
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)


@app.patch("/api/trades/<trade_id>")
def api_trades_update(trade_id: str):
    b   = request.get_json() or {}
    msg = update_trade_fields(
        trade_id,
        b.get("strategy_name", ""),
        b.get("notes", ""),
        b.get("tags", ""),
    )
    return jsonify({"ok": True, "message": msg})


@app.delete("/api/trades/<trade_id>")
def api_trades_delete(trade_id: str):
    msg = delete_trade(trade_id)
    return jsonify({"ok": True, "message": msg})


@app.post("/api/trades/refresh-prices")
def api_trades_refresh():
    msg = refresh_open_prices()
    return jsonify({"ok": True, "message": msg})


# ═══════════════════════════════════════════════════════════════════════════════
# New Finds — social media & recommendation tracker
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/new-finds")
def new_finds_panel():
    return render_template("new_finds_panel.html",
                           sources=SOURCES, categories=CATEGORIES, sentiments=SENTIMENTS,
                           current_week=_current_week())


@app.get("/api/new-finds")
def api_new_finds_list():
    week   = request.args.get("week", "")
    status = request.args.get("status", "")
    return jsonify(get_finds(week, status))


@app.get("/api/new-finds/weeks")
def api_new_finds_weeks():
    return jsonify(get_weeks())


@app.post("/api/new-finds")
def api_new_finds_add():
    b = request.get_json() or {}
    result = add_find(
        b.get("ticker", ""),
        b.get("name", ""),
        b.get("source", ""),
        b.get("category", "Stock"),
        b.get("sentiment", "Watching"),
        b.get("why", ""),
        b.get("link", ""),
        b.get("week_of", ""),
    )
    return jsonify(result), (200 if result["ok"] else 400)


@app.patch("/api/new-finds/<find_id>/status")
def api_new_finds_status(find_id: str):
    b      = request.get_json() or {}
    result = update_find_status(find_id, b.get("status", ""))
    return jsonify(result), (200 if result["ok"] else 400)


@app.delete("/api/new-finds/<find_id>")
def api_new_finds_delete(find_id: str):
    result = delete_find(find_id)
    return jsonify(result), (200 if result["ok"] else 400)


@app.post("/api/new-finds/<find_id>/analyze")
def api_new_finds_analyze(find_id: str):
    force  = (request.get_json() or {}).get("force", False)
    result = analyze_find(find_id, force=force)
    return jsonify(result), (200 if result["ok"] else 400)


@app.get("/api/new-finds/recommendations")
def api_new_finds_recommendations():
    force  = request.args.get("force", "false").lower() == "true"
    result = get_weekly_recommendations(force=force)
    return jsonify(result), (200 if result["ok"] else 500)


@app.post("/api/new-finds/recommendations/reset")
def api_new_finds_reco_reset():
    """Clear in-memory reco cache + error state — forces fresh fetch on next load."""
    from alphaterminal.new_finds import _reco_cache, _reco_status
    _reco_cache.clear()
    _reco_status["error"]   = None
    _reco_status["running"] = False
    return jsonify({"ok": True, "message": "Cache cleared — next load will fetch fresh."})


@app.get("/api/new-finds/recommendations/status")
def api_new_finds_reco_status():
    from alphaterminal.new_finds import _reco_status, _reco_cache
    return jsonify({
        "running":      _reco_status["running"],
        "error":        _reco_status["error"],
        "has_results":  bool(_reco_cache.get("items")),
        "generated_at": _reco_cache["generated_at"].isoformat() if _reco_cache.get("generated_at") else None,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# AlgoBot Performance Report
# ═══════════════════════════════════════════════════════════════════════════════
from alphaterminal.algobot_report import get_summary, get_day_trades, BOTS


@app.get("/algobot")
def algobot_panel():
    return render_template("algobot_panel.html", bots=BOTS)


@app.get("/api/algobot/<bot_key>/summary")
def api_algobot_summary(bot_key: str):
    result = get_summary(bot_key)
    return jsonify(result), (200 if result["ok"] else 400)


@app.get("/api/algobot/<bot_key>/day/<date_str>")
def api_algobot_day(bot_key: str, date_str: str):
    result = get_day_trades(bot_key, date_str)
    return jsonify(result), (200 if result["ok"] else 400)


@app.get("/api/algobot/orders/pending")
def api_pending_orders():
    result = get_pending_orders()
    return jsonify(result), (200 if result["ok"] else 400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
