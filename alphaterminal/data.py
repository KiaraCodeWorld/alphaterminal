import random
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from ib_insync import Stock, util
from .db import ib


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(0)


def fetch_cnn_fear_greed_index() -> str | None:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent":      "Mozilla/5.0",
        "Accept":          "application/json",
        "Referer":         "https://edition.cnn.com/markets/fear-and-greed",
        "Origin":          "https://edition.cnn.com",
    }
    try:
        r    = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        fng  = r.json().get("fear_and_greed") or {}
        score, rating, ts = fng.get("score"), fng.get("rating"), fng.get("timestamp")
        if score is None:
            return None
        val  = int(round(float(score)))
        rtxt = rating.title() if isinstance(rating, str) else "Unknown"
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            ts = str(ts) if ts else "unknown time"
        return f"{val} ({rtxt}) — {ts}"
    except Exception:
        return None


def fetch_fear_greed_index() -> str:
    cnn = fetch_cnn_fear_greed_index()
    if cnn:
        return f"<strong>CNN Fear &amp; Greed Index</strong>: {cnn}"
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=10)
        r.raise_for_status()
        item = r.json()["data"][0]
        date = datetime.utcfromtimestamp(int(item.get("timestamp", time.time()))).strftime("%Y-%m-%d")
        return f"<strong>Fear &amp; Greed Index</strong>: {item['value']} ({item['value_classification']}) — {date}"
    except Exception as exc:
        return f"<strong>Fear &amp; Greed Index</strong>: unavailable ({exc})"


def fetch_yfinance_bars(symbol: str, days: int = 60) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        yf_sym = symbol.replace(".", "-").upper()
        df = yf.Ticker(yf_sym).history(period=f"{days + 10}d", interval="1d", auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df if {"close", "volume"}.issubset(df.columns) else None
    except Exception:
        return None


def fetch_historic_bars(symbol: str, days: int = 60) -> pd.DataFrame | None:
    if not ib.isConnected():
        try:
            ib.connect("127.0.0.1", 7497, clientId=random.randint(1000, 9999))
        except Exception:
            pass

    if ib.isConnected():
        contract = Stock(symbol, "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract, endDateTime="", durationStr=f"{days + 5} D",
                barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1,
            )
            if bars:
                df = util.df(bars)
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as exc:
            print(f"⚠️  IBKR bars failed for {symbol}: {exc}")

    fb = fetch_yfinance_bars(symbol, days)
    if fb is not None and len(fb) >= 20:
        return fb
    return None


def fetch_market_data(symbols: list[str], normalize_fn) -> dict[str, dict]:
    from .utils import tradingview_url
    results = {}
    for symbol in symbols:
        norm = normalize_fn(symbol)
        df   = fetch_historic_bars(norm, days=60)
        if df is None or len(df) < 20:
            results[symbol] = {"error": "Insufficient data"}
            continue

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)
        rsi    = calc_rsi(close)

        lc  = float(close.iloc[-1])
        pc  = float(close.iloc[-2]) if len(close) >= 2 else lc
        c1  = ((lc - pc) / pc * 100) if pc else 0.0
        c7  = ((lc - float(close.iloc[-6])) / float(close.iloc[-6]) * 100) if len(close) >= 7 else 0.0
        c30 = ((lc - float(close.iloc[-26])) / float(close.iloc[-26]) * 100) if len(close) >= 27 else 0.0

        avg_vol  = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())

        results[symbol] = {
            "price":               lc,
            "change_pct":          c1,
            "change_7d":           c7,
            "change_30d":          c30,
            "rsi":                 float(rsi.iloc[-1]) if not rsi.empty else 0.0,
            "below_volume_profile": float(volume.iloc[-1]) < avg_vol,
            "support":             float(close.tail(20).min()),
            "resistance":          float(close.tail(20).max()),
            "tradingview_url":     tradingview_url(symbol),
        }
    return results
