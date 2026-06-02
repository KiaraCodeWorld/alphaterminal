"""
FIRE & Retirement portfolio tracker.
Portfolios live in `fire_portfolios`, holdings in `fire_holdings`.
"""
import concurrent.futures
from datetime import datetime
from bson import ObjectId

import yfinance as yf

from .db import fire_portfolios_collection, fire_holdings_collection
from .data import calc_rsi

# ── Curated dividend / ETF suggestions ───────────────────────────────────────
# Each entry carries a category and a thesis note shown in the UI.
# Live data (price, RSI, yield, opp score) is fetched at request time.

_SUGGESTIONS = [
    # Core ETF starters
    {"symbol": "SCHD",  "category": "Core ETF",         "thesis": "Dividend growth + quality screen; most popular div ETF"},
    {"symbol": "VYM",   "category": "Core ETF",         "thesis": "Broad high-yield exposure, Vanguard low-cost"},
    {"symbol": "DGRO",  "category": "Growth ETF",       "thesis": "iShares dividend growth; 10-yr increase requirement"},
    {"symbol": "VIG",   "category": "Growth ETF",       "thesis": "Vanguard dividend appreciation; consecutive-raise screen"},
    {"symbol": "JEPI",  "category": "Income ETF",       "thesis": "Monthly income via covered calls; smoothed equity exposure"},
    {"symbol": "DIVO",  "category": "Income ETF",       "thesis": "Active covered call on blue chips; premium + dividend"},
    # REITs
    {"symbol": "O",     "category": "Monthly REIT",     "thesis": "30-yr dividend history; paid monthly; Dividend Aristocrat"},
    {"symbol": "STAG",  "category": "Monthly REIT",     "thesis": "Industrial REIT; monthly dividend; e-commerce tailwind"},
    # BDCs
    {"symbol": "MAIN",  "category": "BDC",              "thesis": "Monthly + special dividends; internally managed BDC"},
    {"symbol": "ARCC",  "category": "BDC",              "thesis": "Largest BDC by assets; consistent high yield since 2004"},
    # Dividend Kings / Aristocrats
    {"symbol": "KO",    "category": "Dividend King",    "thesis": "62+ consecutive years of raises; defensive moat"},
    {"symbol": "JNJ",   "category": "Dividend King",    "thesis": "Healthcare giant; 60+ yr raises; spun off Kenvue"},
    {"symbol": "PG",    "category": "Dividend King",    "thesis": "Consumer staples moat; 60+ yr increases"},
    {"symbol": "ABBV",  "category": "Aristocrat",       "thesis": "Post-Humira diversified; aesthetics + immunology pipeline"},
    {"symbol": "MCD",   "category": "Aristocrat",       "thesis": "Franchise model; pricing power; global scale"},
    # High-yield income
    {"symbol": "T",     "category": "High Yield",       "thesis": "Post-spinoff dividend stabilized; 5G infrastructure play"},
    {"symbol": "VZ",    "category": "High Yield",       "thesis": "Telecom yield; network capex cycle peaking"},
    {"symbol": "MO",    "category": "High Yield",       "thesis": "Highest-yield aristocrat; pricing power; smoke-free pivot"},
    # Dividend growth compounders
    {"symbol": "HD",    "category": "Dividend Growth",  "thesis": "Home improvement cycle; fastest-growing div in retail"},
    {"symbol": "MSFT",  "category": "Dividend Growth",  "thesis": "Low yield but fastest-growing; cloud + AI tailwind"},
    {"symbol": "V",     "category": "Dividend Growth",  "thesis": "Duopoly payments; low yield but compounding machine"},
    {"symbol": "AVGO",  "category": "Dividend Growth",  "thesis": "Semiconductor + software; rapid dividend growth since 2010"},
]

_CATEGORY_ORDER = [
    "Core ETF", "Growth ETF", "Income ETF",
    "Monthly REIT", "BDC",
    "Dividend King", "Aristocrat", "High Yield", "Dividend Growth",
]


def get_suggestions() -> list[dict]:
    """
    Enrich the curated suggestion list with live market data.
    Returns list sorted by opportunity score desc within each category.
    """
    symbols = [s["symbol"] for s in _SUGGESTIONS]
    meta    = {s["symbol"]: s for s in _SUGGESTIONS}

    enriched: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        fs = {ex.submit(enrich_symbol, sym): sym for sym in symbols}
        for f in concurrent.futures.as_completed(fs):
            enriched[fs[f]] = f.result()

    results = []
    for sym in symbols:
        live = enriched.get(sym, {})
        m    = meta[sym]

        # "Buy now" label based on live technicals
        rsi   = live.get("rsi")
        opp   = live.get("opp_score", 0)
        dy    = live.get("div_yield") or 0
        rng   = live.get("range_pct")

        if opp >= 7:
            buy_flag = "strong_buy"
        elif opp >= 5:
            buy_flag = "buy"
        elif rsi and rsi < 35:
            buy_flag = "oversold"
        elif rng is not None and rng <= 25:
            buy_flag = "near_low"
        else:
            buy_flag = None

        results.append({
            **live,
            "symbol":   sym,
            "category": m["category"],
            "thesis":   m["thesis"],
            "buy_flag": buy_flag,
        })

    # Sort: within each category by opp_score desc
    cat_order = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    results.sort(key=lambda x: (
        cat_order.get(x["category"], 99),
        -(x.get("opp_score") or 0),
    ))
    return results

# ── Default portfolios (seeded on first run) ──────────────────────────────────

_DEFAULTS = [
    {"name": "FIRE Strategy",      "type": "FIRE",     "color": "#1D9E75",
     "description": "Growth-focused portfolio for financial independence & early retirement."},
    {"name": "Dividend Income",    "type": "DIVIDEND", "color": "#f59e0b",
     "description": "High-yield dividend stocks for reliable passive income."},
    {"name": "AI-Focused",         "type": "AI",       "color": "#7c6af7",
     "description": "AI and technology growth stocks for high upside."},
    {"name": "ETF-Only Portfolio", "type": "ETF",      "color": "#60a5fa",
     "description": "Broad market ETFs — low cost, passive, long-term DCA."},
]

if fire_portfolios_collection.count_documents({}) == 0:
    now = datetime.now()
    fire_portfolios_collection.insert_many([{**p, "created_at": now, "updated_at": now} for p in _DEFAULTS])

# ── Opportunity scoring ───────────────────────────────────────────────────────

def calc_opportunity_score(rsi, price, high52, target_upside_pct, div_yield_pct) -> int:
    score = 0

    # RSI — the closer to oversold the better (0-3 pts)
    if rsi is not None:
        if rsi < 30:   score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1

    # Distance below 52-week high — deeper pullback = more opportunity (0-3 pts)
    if price and high52 and high52 > 0:
        pct_below = (high52 - price) / high52 * 100
        if pct_below >= 35:   score += 3
        elif pct_below >= 20: score += 2
        elif pct_below >= 10: score += 1

    # Analyst target upside (0-3 pts)
    if target_upside_pct is not None:
        if target_upside_pct >= 30:   score += 3
        elif target_upside_pct >= 15: score += 2
        elif target_upside_pct >= 5:  score += 1

    # Dividend yield bonus — rewards high-yield entries (0-1 pt)
    if div_yield_pct and div_yield_pct >= 3:
        score += 1

    return min(score, 10)


def _buy_signal_label(portfolio_type: str, data: dict) -> str | None:
    """Return a strategy-specific buy signal label, or None."""
    rsi        = data.get("rsi")
    div_yield  = data.get("div_yield") or 0
    range_pct  = data.get("range_pct")     # 0=at 52w low, 100=at 52w high
    opp        = data.get("opp_score", 0)
    upside     = data.get("target_upside") or 0
    payout     = data.get("payout_ratio") or 1

    if portfolio_type == "FIRE":
        if opp >= 7: return "Strong Accumulate"
        if opp >= 5: return "Watch & DCA"
    elif portfolio_type == "DIVIDEND":
        if div_yield >= 4 and rsi and rsi < 50 and payout < 0.8:
            return "High-Yield Entry"
        if div_yield >= 2.5 and opp >= 5:
            return "Div DCA Zone"
    elif portfolio_type == "AI":
        if range_pct is not None and range_pct <= 35 and rsi and rsi < 45:
            return "Tech Dip Entry"
        if upside >= 25: return "High Upside"
    elif portfolio_type == "ETF":
        if range_pct is not None and range_pct <= 30:
            return "DCA Into Dip"
        if rsi and rsi < 40:
            return "DCA Zone"
    return None


# ── Market data enrichment ────────────────────────────────────────────────────

def enrich_symbol(symbol: str) -> dict:
    try:
        tk   = yf.Ticker(symbol.replace(".", "-"))
        info = tk.info or {}
        hist = tk.history(period="1y")

        price   = info.get("regularMarketPrice") or info.get("currentPrice")
        prev    = info.get("previousClose")
        high52  = info.get("fiftyTwoWeekHigh")
        low52   = info.get("fiftyTwoWeekLow")
        target  = info.get("targetMeanPrice")
        div_raw = info.get("dividendYield") or 0
        div_yield = round(div_raw * 100, 2) if div_raw else None

        change_pct = round((price - prev) / prev * 100, 2) if price and prev else None

        # RSI from 1-year daily closes
        rsi = None
        if not hist.empty and "Close" in hist.columns:
            rsi_s = calc_rsi(hist["Close"])
            if not rsi_s.empty:
                rsi = round(float(rsi_s.iloc[-1]), 1)

        # Position in 52-week range (0% = at low, 100% = at high)
        range_pct = None
        if price and high52 and low52 and high52 > low52:
            range_pct = round((price - low52) / (high52 - low52) * 100, 1)

        target_upside = round((target - price) / price * 100, 1) if target and price else None

        opp = calc_opportunity_score(rsi, price, high52, target_upside, div_yield)

        return {
            "name":          info.get("longName") or info.get("shortName") or symbol,
            "sector":        info.get("sector", ""),
            "price":         price,
            "change_pct":    change_pct,
            "high_52w":      high52,
            "low_52w":       low52,
            "range_pct":     range_pct,
            "target":        target,
            "target_upside": target_upside,
            "rsi":           rsi,
            "div_yield":     div_yield,
            "div_rate":      info.get("dividendRate"),
            "payout_ratio":  info.get("payoutRatio"),
            "forward_pe":    round(info.get("forwardPE", 0), 1) or None,
            "market_cap":    info.get("marketCap"),
            "rec_key":       info.get("recommendationKey", ""),
            "opp_score":     opp,
        }
    except Exception as exc:
        return {"error": str(exc), "price": None, "opp_score": 0, "name": symbol}


# ── Portfolio CRUD ────────────────────────────────────────────────────────────

def get_portfolios() -> list[dict]:
    out = []
    for doc in fire_portfolios_collection.find().sort("created_at", 1):
        count = fire_holdings_collection.count_documents({"portfolio_id": str(doc["_id"])})
        out.append({
            "id":          str(doc["_id"]),
            "name":        doc["name"],
            "type":        doc.get("type", "CUSTOM"),
            "color":       doc.get("color", "#7a8499"),
            "description": doc.get("description", ""),
            "count":       count,
        })
    return out


def create_portfolio(name: str, ptype: str, description: str, color: str) -> str:
    result = fire_portfolios_collection.insert_one({
        "name":        name.strip(),
        "type":        ptype.upper(),
        "color":       color or "#7a8499",
        "description": description.strip(),
        "created_at":  datetime.now(),
        "updated_at":  datetime.now(),
    })
    return str(result.inserted_id)


def delete_portfolio(portfolio_id: str) -> None:
    fire_portfolios_collection.delete_one({"_id": ObjectId(portfolio_id)})
    fire_holdings_collection.delete_many({"portfolio_id": portfolio_id})


# ── Holdings CRUD ─────────────────────────────────────────────────────────────

def add_holding(portfolio_id: str, symbol: str, status: str = "watching",
                target_alloc_pct: float | None = None, dca_target_price: float | None = None,
                shares_owned: float | None = None, avg_cost: float | None = None,
                notes: str = "", tags: list[str] | None = None) -> str:
    symbol = symbol.strip().upper()
    existing = fire_holdings_collection.find_one({"portfolio_id": portfolio_id, "symbol": symbol})
    if existing:
        return str(existing["_id"])
    result = fire_holdings_collection.insert_one({
        "portfolio_id":     portfolio_id,
        "symbol":           symbol,
        "status":           status,
        "target_alloc_pct": target_alloc_pct,
        "dca_target_price": dca_target_price,
        "shares_owned":     shares_owned,
        "avg_cost":         avg_cost,
        "notes":            notes.strip(),
        "tags":             tags or [],
        "added_at":         datetime.now(),
        "updated_at":       datetime.now(),
    })
    return str(result.inserted_id)


def remove_holding(portfolio_id: str, symbol: str) -> None:
    fire_holdings_collection.delete_one({"portfolio_id": portfolio_id, "symbol": symbol.upper()})


def update_holding(portfolio_id: str, symbol: str, fields: dict) -> None:
    allowed = {"status", "target_alloc_pct", "dca_target_price",
               "shares_owned", "avg_cost", "notes", "tags"}
    patch = {k: v for k, v in fields.items() if k in allowed}
    patch["updated_at"] = datetime.now()
    fire_holdings_collection.update_one(
        {"portfolio_id": portfolio_id, "symbol": symbol.upper()},
        {"$set": patch},
    )


# ── Holdings with live data ───────────────────────────────────────────────────

def get_holdings_enriched(portfolio_id: str, portfolio_type: str = "") -> list[dict]:
    docs = list(fire_holdings_collection.find({"portfolio_id": portfolio_id}))
    if not docs:
        return []

    enriched: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        fs = {ex.submit(enrich_symbol, d["symbol"]): d["symbol"] for d in docs}
        for f in concurrent.futures.as_completed(fs):
            enriched[fs[f]] = f.result()

    result = []
    for doc in docs:
        sym  = doc["symbol"]
        live = enriched.get(sym, {})
        merged = {
            "id":               str(doc["_id"]),
            "portfolio_id":     portfolio_id,
            "symbol":           sym,
            "status":           doc.get("status", "watching"),
            "target_alloc_pct": doc.get("target_alloc_pct"),
            "dca_target_price": doc.get("dca_target_price"),
            "shares_owned":     doc.get("shares_owned"),
            "avg_cost":         doc.get("avg_cost"),
            "notes":            doc.get("notes", ""),
            "tags":             doc.get("tags", []),
            "added_at":         doc["added_at"].strftime("%m/%d/%y") if doc.get("added_at") else None,
            **live,
        }
        merged["buy_signal"] = _buy_signal_label(portfolio_type, merged)

        # Unrealized P&L if owned
        price = live.get("price")
        if price and doc.get("shares_owned") and doc.get("avg_cost"):
            merged["unrealized_pnl"]     = round((price - doc["avg_cost"]) * doc["shares_owned"], 2)
            merged["unrealized_pnl_pct"] = round((price - doc["avg_cost"]) / doc["avg_cost"] * 100, 2)
            merged["position_value"]     = round(price * doc["shares_owned"], 2)
        else:
            merged["unrealized_pnl"]     = None
            merged["unrealized_pnl_pct"] = None
            merged["position_value"]     = None

        result.append(merged)

    result.sort(key=lambda x: x.get("opp_score", 0), reverse=True)
    return result
