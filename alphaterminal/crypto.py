"""
Crypto watchlist — stores coins in `crypto_watchlist`, enriches with live yfinance data.
"""
import concurrent.futures
from datetime import datetime

import yfinance as yf

from .db import crypto_collection
from .data import calc_rsi

_DEFAULTS = [
    {"symbol": "BTC",  "name": "Bitcoin",  "yf_ticker": "BTC-USD",  "color": "#f7931a"},
    {"symbol": "ETH",  "name": "Ethereum", "yf_ticker": "ETH-USD",  "color": "#627eea"},
    {"symbol": "SOL",  "name": "Solana",   "yf_ticker": "SOL-USD",  "color": "#9945ff"},
    {"symbol": "XRP",  "name": "XRP",      "yf_ticker": "XRP-USD",  "color": "#346aa9"},
]

if crypto_collection.count_documents({}) == 0:
    now = datetime.now()
    crypto_collection.insert_many([{**c, "added_at": now} for c in _DEFAULTS])


def _yf_ticker(symbol: str) -> str:
    doc = crypto_collection.find_one({"symbol": symbol.upper()})
    return doc["yf_ticker"] if doc else f"{symbol.upper()}-USD"


def enrich_coin(symbol: str) -> dict:
    try:
        ticker_str = _yf_ticker(symbol)
        tk   = yf.Ticker(ticker_str)
        info = tk.info or {}
        hist = tk.history(period="1y")

        price  = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("open")
        prev   = info.get("previousClose")
        high52 = info.get("fiftyTwoWeekHigh")
        low52  = info.get("fiftyTwoWeekLow")

        change_pct = round((price - prev) / prev * 100, 2) if price and prev else None

        # Market cap / volume
        market_cap = info.get("marketCap")
        volume_24h = info.get("volume24Hr") or info.get("volume")

        # RSI
        rsi = None
        if not hist.empty and "Close" in hist.columns:
            rsi_s = calc_rsi(hist["Close"])
            if not rsi_s.empty:
                rsi = round(float(rsi_s.iloc[-1]), 1)

        # 52W position (0% = at low, 100% = at high)
        range_pct = None
        if price and high52 and low52 and high52 > low52:
            range_pct = round((price - low52) / (high52 - low52) * 100, 1)

        # 7-day and 30-day change from history
        change_7d = change_30d = None
        if not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 7:
                c7 = float(closes.iloc[-7])
                change_7d = round((float(closes.iloc[-1]) - c7) / c7 * 100, 2) if c7 else None
            if len(closes) >= 30:
                c30 = float(closes.iloc[-30])
                change_30d = round((float(closes.iloc[-1]) - c30) / c30 * 100, 2) if c30 else None

        return {
            "price":      price,
            "change_pct": change_pct,
            "change_7d":  change_7d,
            "change_30d": change_30d,
            "high_52w":   high52,
            "low_52w":    low52,
            "range_pct":  range_pct,
            "market_cap": market_cap,
            "volume_24h": volume_24h,
            "rsi":        rsi,
        }
    except Exception as exc:
        return {"error": str(exc), "price": None}


def get_coins_enriched() -> list[dict]:
    docs = list(crypto_collection.find().sort("added_at", 1))
    if not docs:
        return []

    enriched: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        fs = {ex.submit(enrich_coin, d["symbol"]): d["symbol"] for d in docs}
        for f in concurrent.futures.as_completed(fs):
            enriched[fs[f]] = f.result()

    result = []
    for doc in docs:
        sym  = doc["symbol"]
        live = enriched.get(sym, {})
        result.append({
            "symbol":     sym,
            "name":       doc.get("name", sym),
            "yf_ticker":  doc.get("yf_ticker", f"{sym}-USD"),
            "color":      doc.get("color", "#7a8499"),
            "added_at":   doc["added_at"].strftime("%m/%d/%y") if doc.get("added_at") else None,
            **live,
        })
    return result


def add_coin(symbol: str, name: str = "", color: str = "#7a8499") -> str:
    symbol = symbol.strip().upper()
    existing = crypto_collection.find_one({"symbol": symbol})
    if existing:
        return str(existing["_id"])
    yf_ticker = f"{symbol}-USD"
    result = crypto_collection.insert_one({
        "symbol":    symbol,
        "name":      name.strip() or symbol,
        "yf_ticker": yf_ticker,
        "color":     color,
        "added_at":  datetime.now(),
    })
    return str(result.inserted_id)


def remove_coin(symbol: str) -> None:
    # Prevent removing defaults
    defaults = {d["symbol"] for d in _DEFAULTS}
    if symbol.upper() in defaults:
        return
    crypto_collection.delete_one({"symbol": symbol.upper()})
