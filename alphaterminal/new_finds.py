import os
import json
import threading
from datetime import datetime
from bson import ObjectId
from openai import OpenAI
from .db import new_finds_collection


SOURCES    = ["Reddit", "Twitter/X", "StockTwits", "YouTube", "Newsletter", "Discord", "TikTok", "Manual"]
CATEGORIES = ["Stock", "ETF", "Crypto", "Option Play", "Sector Theme"]
SENTIMENTS = ["Bullish", "Neutral", "Bearish", "Watching"]

_analyze_lock = threading.Lock()


# ── DeepSeek client (OpenAI-compatible) ──────────────────────────────────────

def _openai_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip().strip("\"'")
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set.")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


# ── yfinance snapshot ─────────────────────────────────────────────────────────

def _fetch_market_data(ticker: str) -> dict:
    try:
        import yfinance as yf
        obj  = yf.Ticker(ticker.replace(".", "-"))
        info = obj.info or {}
    except Exception:
        return {}

    price = info.get("regularMarketPrice") or info.get("currentPrice")
    prev  = info.get("previousClose")
    change_pct = round((price - prev) / prev * 100, 2) if price and prev else None

    try:
        news = obj.news or []
    except Exception:
        news = []
    headlines = [n.get("title", "") for n in news if n.get("title")][:6]

    analyst_parts = []
    if info.get("targetMeanPrice"):      analyst_parts.append(f"Mean Target: ${info['targetMeanPrice']:.2f}")
    if info.get("targetHighPrice"):      analyst_parts.append(f"High Target: ${info['targetHighPrice']:.2f}")
    if info.get("targetLowPrice"):       analyst_parts.append(f"Low Target: ${info['targetLowPrice']:.2f}")
    if info.get("numberOfAnalystOpinions"): analyst_parts.append(f"Analysts: {info['numberOfAnalystOpinions']}")
    if info.get("recommendationKey"):    analyst_parts.append(f"Consensus: {info['recommendationKey']}")
    if info.get("shortPercentOfFloat"):  analyst_parts.append(f"Short Float: {info['shortPercentOfFloat']*100:.1f}%")

    mc = info.get("marketCap")

    return {
        "price":            price,
        "change_pct":       change_pct,
        "volume":           info.get("volume"),
        "avg_volume":       info.get("averageVolume"),
        "market_cap":       mc,
        "market_cap_str":   (f"${mc/1e12:.2f}T" if mc and mc>=1e12 else f"${mc/1e9:.1f}B" if mc and mc>=1e9 else f"${mc/1e6:.0f}M" if mc else "N/A"),
        "52wk_high":        info.get("fiftyTwoWeekHigh"),
        "52wk_low":         info.get("fiftyTwoWeekLow"),
        "pe_ratio":         info.get("trailingPE"),
        "sector":           info.get("sector", ""),
        "industry":         info.get("industry", ""),
        "short_name":       info.get("shortName", ""),
        "analyst_summary":  " | ".join(analyst_parts),
        "news_headlines":   headlines,
    }


# ── OpenAI prompt ─────────────────────────────────────────────────────────────

def _build_analysis_prompt(find: dict, mkt: dict) -> str:
    headlines = "\n".join(f"- {h}" for h in mkt.get("news_headlines", [])) or "- No recent news available"
    return f"""You are a professional trader and analyst evaluating a social-media-discovered stock/asset.

TICKER: {find['ticker']}
FULL NAME: {mkt.get('short_name') or find.get('name', 'Unknown')}
SECTOR: {mkt.get('sector', 'Unknown')} / {mkt.get('industry', '')}
DISCOVERED ON: {find.get('source', 'Social media')}
COMMUNITY THESIS: {find.get('why') or 'Not provided'}
COMMUNITY SENTIMENT: {find.get('sentiment', 'Unknown')}
CATEGORY: {find.get('category', 'Stock')}

LIVE MARKET DATA:
- Price: ${mkt.get('price', 'N/A')}
- Daily Change: {mkt.get('change_pct', 'N/A')}%
- Volume: {mkt.get('volume', 'N/A')} (Avg: {mkt.get('avg_volume', 'N/A')})
- Market Cap: {mkt.get('market_cap_str', 'N/A')}
- 52-Week High: ${mkt.get('52wk_high', 'N/A')} | 52-Week Low: ${mkt.get('52wk_low', 'N/A')}
- P/E Ratio: {mkt.get('pe_ratio', 'N/A')}
- Analyst Data: {mkt.get('analyst_summary') or 'Not available'}

RECENT NEWS:
{headlines}

Your job: evaluate whether this is a genuine opportunity or social media noise.

Return ONLY a valid JSON object — no markdown fences, no extra text — with EXACTLY these fields:

{{
  "verdict": "STRONG_BUY" | "BUY_DIP" | "WATCH" | "AVOID",
  "verdict_reason": "1 sentence summary of the core thesis or why to avoid",
  "price_now": <number or null>,
  "price_30d_target": <number — your 30-day price estimate>,
  "price_90d_target": <number — your 90-day price estimate>,
  "upside_pct_30d": <number — % upside to 30d target>,
  "upside_pct_90d": <number — % upside to 90d target>,
  "bull_case": ["point 1", "point 2", "point 3"],
  "bear_case": ["point 1", "point 2", "point 3"],
  "main_catalyst": "the key driver in 1 sentence",
  "key_support": <number or null>,
  "key_resistance": <number or null>,
  "entry_zone": "e.g. $45–48 on pullback",
  "stop_loss": <number or null>,
  "target_price": <number or null>,
  "risk_reward_ratio": <number or null>,
  "social_hype_real": true | false,
  "hype_assessment": "1-2 sentences: is community excitement backed by fundamentals/catalysts?",
  "sector_tailwind": "is the broader sector supportive? 1 sentence",
  "volume_signal": "what does volume tell us? 1 sentence",
  "confidence": <integer 1-10>,
  "confidence_reason": "1 sentence why you're this confident or not"
}}"""


# ── Analyze a find ────────────────────────────────────────────────────────────

def analyze_find(find_id: str, force: bool = False) -> dict:
    """Run OpenAI analysis on a find. Returns the analysis dict."""
    try:
        doc = new_finds_collection.find_one({"_id": ObjectId(find_id)})
    except Exception:
        return {"ok": False, "error": "Invalid find ID."}
    if not doc:
        return {"ok": False, "error": "Find not found."}

    # Return cached result unless force=True (cached for 24h)
    cached = doc.get("ai_analysis")
    cached_at = doc.get("ai_analyzed_at")
    if not force and cached and cached_at:
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        if age_hours < 24:
            return {"ok": True, "analysis": cached, "from_cache": True}

    ticker = doc["ticker"]
    mkt    = _fetch_market_data(ticker)

    try:
        client = _openai_client()
        resp   = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional stock trader and analyst. Return only valid JSON as instructed. No markdown, no explanation."},
                {"role": "user",   "content": _build_analysis_prompt(doc, mkt)},
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=40,
        )
        raw  = resp.choices[0].message.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analysis = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"JSON parse error: {exc}", "raw": raw if 'raw' in dir() else ""}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    # Attach live market data to the response
    analysis["_market"] = {
        "price":        mkt.get("price"),
        "change_pct":   mkt.get("change_pct"),
        "market_cap":   mkt.get("market_cap_str"),
        "52wk_high":    mkt.get("52wk_high"),
        "52wk_low":     mkt.get("52wk_low"),
        "sector":       mkt.get("sector"),
        "short_name":   mkt.get("short_name"),
    }

    # Cache on the find document
    new_finds_collection.update_one(
        {"_id": ObjectId(find_id)},
        {"$set": {
            "ai_analysis":    analysis,
            "ai_analyzed_at": datetime.now(),
            "updated_at":     datetime.now(),
        }},
    )

    return {"ok": True, "analysis": analysis, "from_cache": False}


# ── Weekly AI recommendations ─────────────────────────────────────────────────

_reco_cache:    dict = {}   # {"generated_at": datetime, "items": [...]}
_reco_status:   dict = {"running": False, "error": None}
_RECO_TTL_HOURS = 1


def _run_recommendations() -> None:
    """Background worker — populates _reco_cache."""
    _reco_status["running"] = True
    _reco_status["error"]   = None
    try:
        result = _fetch_recommendations_sync()
        if result["ok"]:
            _reco_cache["items"]        = result["items"]
            _reco_cache["generated_at"] = datetime.now()
        else:
            _reco_status["error"] = result.get("error", "Unknown error")
    except Exception as exc:
        _reco_status["error"] = str(exc)
    finally:
        _reco_status["running"] = False


def get_weekly_recommendations(force: bool = False) -> dict:
    """Returns cached recommendations instantly, or kicks off a background job."""
    now = datetime.now()

    # Return cache if still fresh
    if not force and _reco_cache.get("generated_at"):
        age = (now - _reco_cache["generated_at"]).total_seconds() / 3600
        if age < _RECO_TTL_HOURS:
            return {"ok": True, "items": _reco_cache["items"], "from_cache": True,
                    "generated_at": _reco_cache["generated_at"].isoformat()}

    # Already generating — tell frontend to poll
    if _reco_status["running"]:
        return {"ok": True, "items": _reco_cache.get("items", []),
                "status": "generating", "from_cache": bool(_reco_cache.get("items"))}

    # Last run errored — surface it
    if _reco_status["error"] and not force:
        return {"ok": False, "error": _reco_status["error"]}

    # Start background generation
    threading.Thread(target=_run_recommendations, daemon=True).start()
    return {"ok": True, "items": [], "status": "generating", "from_cache": False}


def _fetch_recommendations_sync() -> dict:
    """Blocking — do the actual DeepSeek call. Called only from background thread."""
    now = datetime.now()

    today_str = now.strftime("%B %d, %Y")
    prompt = f"""Today is {today_str}. You are a professional trader who tracks social media (Reddit WSB, Twitter/X, StockTwits, YouTube) and news daily.

Pick the TOP 4 US-listed stocks or ETFs generating the most genuine buzz and opportunity this week. Choose a mix: 1-2 momentum plays, 1 under-the-radar catalyst, 1 sector theme or contrarian.

For EACH ticker return a COMPLETE analysis. Return ONLY a valid JSON array of 4 objects — no markdown, no extra text:

[
  {{
    "ticker": "SYMBOL",
    "name": "Full Company Name",
    "source": "Reddit" | "Twitter/X" | "StockTwits" | "YouTube" | "Newsletter",
    "category": "Stock" | "ETF" | "Crypto" | "Option Play",
    "sentiment": "Bullish" | "Bearish" | "Watching",
    "why": "1-2 sentences why the community is excited right now",
    "catalyst": "specific catalyst in 1 sentence",
    "verdict": "STRONG_BUY" | "BUY_DIP" | "WATCH" | "AVOID",
    "verdict_reason": "1 sentence core thesis",
    "price_30d_target": <number>,
    "price_90d_target": <number>,
    "upside_pct_30d": <number>,
    "upside_pct_90d": <number>,
    "bull_case": ["point 1", "point 2"],
    "bear_case": ["point 1", "point 2"],
    "entry_zone": "e.g. $45-48 on pullback",
    "stop_loss": <number or null>,
    "target_price": <number or null>,
    "risk_reward_ratio": <number or null>,
    "key_support": <number or null>,
    "key_resistance": <number or null>,
    "social_hype_real": true | false,
    "hype_assessment": "1 sentence: is excitement backed by real catalyst?",
    "sector_tailwind": "1 sentence on sector support",
    "confidence": <integer 1-10>,
    "confidence_reason": "1 sentence"
  }}
]"""

    try:
        client = _openai_client()
        resp   = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional stock trader and analyst. Return only valid JSON as instructed. No markdown, no explanation."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1800,
            timeout=40,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        candidates: list = json.loads(raw)
    except Exception as exc:
        return {"ok": False, "error": f"OpenAI call failed: {exc}"}

    # Enrich with live yfinance prices only (no extra API calls)
    items = []
    for c in candidates[:4]:
        ticker = c.get("ticker", "").strip().upper()
        if not ticker:
            continue
        mkt = _fetch_market_data(ticker)
        c["ai_analysis"] = {k: v for k, v in c.items()
                            if k not in ("ticker","name","source","category","sentiment","why","catalyst")}
        c["ai_analysis"]["_market"] = {
            "price":      mkt.get("price"),
            "change_pct": mkt.get("change_pct"),
            "market_cap": mkt.get("market_cap_str"),
            "52wk_high":  mkt.get("52wk_high"),
            "52wk_low":   mkt.get("52wk_low"),
            "sector":     mkt.get("sector"),
            "short_name": mkt.get("short_name") or c.get("name",""),
        }
        items.append({
            "ticker":      ticker,
            "name":        c.get("name", ""),
            "source":      c.get("source", ""),
            "category":    c.get("category", "Stock"),
            "sentiment":   c.get("sentiment", "Watching"),
            "why":         c.get("why", ""),
            "catalyst":    c.get("catalyst", ""),
            "ai_analysis": c["ai_analysis"],
        })

    return {"ok": True, "items": items}


# ── CRUD ──────────────────────────────────────────────────────────────────────

def add_find(ticker: str, name: str, source: str, category: str,
             sentiment: str, why: str, link: str, week_of: str) -> dict:
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ok": False, "error": "Ticker is required."}
    doc = {
        "ticker":       ticker,
        "name":         name.strip(),
        "source":       source.strip(),
        "category":     category.strip(),
        "sentiment":    sentiment.strip(),
        "why":          why.strip(),
        "link":         link.strip(),
        "week_of":      week_of.strip() or _current_week(),
        "status":       "watching",
        "ai_analysis":  None,
        "created_at":   datetime.now(),
        "updated_at":   datetime.now(),
    }
    inserted = new_finds_collection.insert_one(doc)
    return {"ok": True, "id": str(inserted.inserted_id)}


def update_status(find_id: str, status: str) -> dict:
    valid = {"watching", "added_to_watchlist", "passed"}
    if status not in valid:
        return {"ok": False, "error": f"Status must be one of {valid}"}
    try:
        new_finds_collection.update_one(
            {"_id": ObjectId(find_id)},
            {"$set": {"status": status, "updated_at": datetime.now()}},
        )
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def delete_find(find_id: str) -> dict:
    try:
        res = new_finds_collection.delete_one({"_id": ObjectId(find_id)})
        return {"ok": bool(res.deleted_count)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_finds(week_of: str = "", status: str = "") -> list[dict]:
    query: dict = {}
    if week_of:
        query["week_of"] = week_of
    if status:
        query["status"] = status
    docs = list(new_finds_collection.find(query).sort("created_at", -1))
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        for fld in ("created_at", "updated_at", "ai_analyzed_at"):
            if isinstance(d.get(fld), datetime):
                d[fld] = d[fld].isoformat()
        out.append(d)
    return out


def get_weeks() -> list[str]:
    weeks = new_finds_collection.distinct("week_of")
    return sorted(weeks, reverse=True)


def _current_week() -> str:
    return datetime.now().strftime("%Y-W%W")
