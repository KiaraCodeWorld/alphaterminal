import os
import json
import re
import threading
import urllib.parse
from datetime import datetime, timedelta
from openai import OpenAI
from .config import OPENAI_MODEL
from .db import analysis_collection
from .utils import tradingview_url

# ── batch-analysis state ──────────────────────────────────────────────────────
_batch_lock:   threading.Lock          = threading.Lock()
_batch_status: dict[str, str]          = {}  # ticker → "queued"|"done"|"error:…"

_SECTION_ACCENT = {
    "1": "#00d68f", "2": "#00d68f", "3": "#e05a2e",
    "4": "#e05a2e", "5": "#d4a017", "6": "#7c6af7",
    "7": "#7c6af7", "8": "#7c6af7", "9": "#f04040",
    "10": "#00d68f", "11": "#e05a2e",
}


def _openai_client() -> OpenAI:
    key = os.environ.get("OPEN_AI_KEY", "").strip().strip("\"'")
    if not key:
        raise EnvironmentError("OPEN_AI_KEY not set.")
    return OpenAI(api_key=key)


def fetch_stock_data(ticker: str) -> dict:
    import yfinance as yf
    try:
        obj  = yf.Ticker(ticker.replace(".", "-"))
        info = obj.info or {}
    except Exception:
        info = {}

    price = info.get("regularMarketPrice") or info.get("currentPrice")
    prev  = info.get("previousClose")
    change = f"{(price - prev) / prev * 100:+.2f}%" if price and prev else None

    float_s = info.get("floatShares")
    if float_s:
        float_s = f"{float_s:,}"

    try:
        news = obj.news or []
    except Exception:
        news = []
    headlines = [n.get("title", "") for n in news if n.get("title")][:5]

    parts = []
    if info.get("targetMeanPrice"): parts.append(f"Mean Target: ${info['targetMeanPrice']:.2f}")
    if info.get("targetHighPrice"): parts.append(f"High: ${info['targetHighPrice']:.2f}")
    if info.get("targetLowPrice"):  parts.append(f"Low: ${info['targetLowPrice']:.2f}")
    if info.get("numberOfAnalystOpinions"): parts.append(f"Analysts: {info['numberOfAnalystOpinions']}")
    if info.get("recommendationKey"):  parts.append(f"Consensus: {info['recommendationKey'].title()}")
    if info.get("recommendationMean"): parts.append(f"Score: {info['recommendationMean']:.1f}/5")

    return {
        "current_price":       price,
        "daily_change":        change,
        "volume":              info.get("volume"),
        "market_cap":          info.get("marketCap"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low":  info.get("fiftyTwoWeekLow"),
        "float_shares":        float_s,
        "news_summary":        "\n".join(f"- {h}" for h in headlines) or "- No recent headlines",
        "analyst_summary":     "\n".join(parts) or "Not available",
    }


def _build_prompt(ticker: str, data: dict) -> str:
    return (
        f"You are a professional stock market analyst and trader.\n\n"
        f"Analyze the stock ticker: {ticker}\n\n"
        "Generate a SHORT, HIGH-VALUE, TRADER-FOCUSED stock research summary.\n"
        "Return the response in the EXACT structure below:\n\n"
        "1. Company Snapshot\n   - Company Name, Market Cap, Sector, one-line business description\n\n"
        "2. Latest News / Catalysts\n   - Key catalyst and why traders care\n\n"
        "3. Analyst Target\n   - Avg target, upside/downside %\n\n"
        "4. Analyst Recommendations\n   - Consensus rating, Buy/Hold/Sell counts\n\n"
        "5. Recent Analyst Upgrades / Downgrades\n   - Firm, action, target change\n\n"
        "6. Trading Trend\n   - Bullish / Bearish / Sideways (1-2 lines)\n\n"
        "7. Key Levels\n   - Support and Resistance\n\n"
        "8. Momentum & Volume\n   - Volume trend, pattern (breakout/consolidation/squeeze)\n\n"
        "9. Risk Factors\n   - Major risks\n\n"
        "10. Trade Setup Idea\n   - Entry zone, stop loss, target\n\n"
        "11. Confidence Score (1-10)\n   - Score and brief reason\n\n"
        "RULES: concise, bullet points, numbers/%, no disclaimer, say 'Not available' if missing.\n\n"
        f"MARKET DATA: price={data.get('current_price')} change={data.get('daily_change')} "
        f"vol={data.get('volume')} mktcap={data.get('market_cap')} "
        f"52wH={data.get('fifty_two_week_high')} 52wL={data.get('fifty_two_week_low')} "
        f"float={data.get('float_shares')}\n\n"
        f"ANALYST DATA:\n{data.get('analyst_summary')}\n\n"
        f"RECENT NEWS:\n{data.get('news_summary')}\n"
    )


def get_or_fetch_analysis(ticker: str) -> tuple[str, dict]:
    ticker      = ticker.strip().upper()
    cache_limit = datetime.now() - timedelta(days=5)
    cached      = analysis_collection.find_one({"ticker": ticker, "fetched_at": {"$gt": cache_limit}})
    if cached:
        return cached["analysis"], cached.get("stock_data", {})

    stock_data = fetch_stock_data(ticker)
    try:
        resp = _openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a trading-focused research assistant. Keep answers concise, structured, and actionable."},
                {"role": "user",   "content": _build_prompt(ticker, stock_data)},
            ],
            temperature=0.4,
            max_tokens=1400,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as exc:
        text = f"OpenAI analysis failed: {exc}"

    analysis_collection.update_one(
        {"ticker": ticker},
        {"$set": {"ticker": ticker, "analysis": text, "stock_data": stock_data, "fetched_at": datetime.now()}},
        upsert=True,
    )
    return text, stock_data


def tradingview_embed_html(ticker: str) -> str:
    cfg = json.dumps({"symbol": ticker.upper(), "dateRange": "3M",
                      "colorTheme": "dark", "isTransparent": False, "autosize": True})
    src = f"https://www.tradingview.com/embed-widget/mini-symbol-overview/?locale=en#{urllib.parse.quote(cfg)}"
    tv  = tradingview_url(ticker)
    return (
        f"<div style='margin:12px 0;'>"
        f"<div style='display:flex; justify-content:space-between; margin-bottom:6px;'>"
        f"<span style='color:#94a3b8; font-size:0.85rem;'>TradingView · {ticker}</span>"
        f"<a href='{tv}' target='_blank' style='color:#60a5fa; font-size:0.82rem;'>Open full chart ↗</a></div>"
        f"<iframe src='{src}' width='100%' height='220' frameborder='0' scrolling='no' "
        f"style='border-radius:12px; border:1px solid #1f2937; display:block;'></iframe></div>"
    )


def _bullets_to_html(text: str) -> str:
    html, in_list = "", False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- ") or line.startswith("• "):
            if not in_list:
                html += "<ul style='margin:6px 0; padding-left:18px;'>"
                in_list = True
            html += f"<li style='margin:3px 0; color:#cbd5e1;'>{line[2:].strip()}</li>"
        else:
            if in_list:
                html += "</ul>"
                in_list = False
            html += f"<p style='margin:4px 0; color:#94a3b8;'>{line}</p>"
    return html + ("</ul>" if in_list else "")


def render_analysis_html(ticker: str, text: str, stock_data: dict) -> str:
    matches  = list(re.finditer(r"^(\d+)\.\s+(.+)$", text, re.MULTILINE))
    sections = [
        {"num": m.group(1), "title": m.group(2).strip(),
         "content": text[m.end(): (matches[i+1].start() if i+1 < len(matches) else len(text))].strip()}
        for i, m in enumerate(matches)
    ]

    price  = stock_data.get("current_price", "N/A")
    change = stock_data.get("daily_change") or ""
    mc     = stock_data.get("market_cap")
    mc_str = (f"${mc/1e12:.2f}T" if mc and mc >= 1e12 else
              f"${mc/1e9:.1f}B"  if mc and mc >= 1e9  else
              f"${mc/1e6:.0f}M"  if mc else "N/A")
    chg_c  = "#22c55e" if "+" in change else "#ef4444"
    tv     = tradingview_url(ticker)

    header = (
        f"<div style='background:#0f172a; border:1px solid #1f2937; border-radius:16px; "
        f"padding:16px 20px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;'>"
        f"<div><h3 style='margin:0; color:#f8fafc;'>"
        f"<a href='{tv}' target='_blank' style='color:#60a5fa; text-decoration:none;'>{ticker} ↗</a></h3>"
        f"<span style='color:#64748b; font-size:0.8rem;'>AI Analysis · cached 5 days</span></div>"
        f"<div style='text-align:right;'>"
        f"<div style='color:#f8fafc; font-size:1.4rem; font-weight:700;'>${price}</div>"
        f"<div style='color:{chg_c}; font-size:0.9rem;'>{change}</div></div>"
        f"<div style='text-align:right;'>"
        f"<div style='color:#94a3b8; font-size:0.78rem;'>MKT CAP</div>"
        f"<div style='color:#f8fafc; font-size:1rem; font-weight:600;'>{mc_str}</div></div></div>"
    )

    if not sections:
        return header + f"<div style='background:#111827; padding:16px; border-radius:12px; color:#cbd5e1;'>{text.replace(chr(10), '<br>')}</div>"

    cards = []
    for s in sections:
        accent = _SECTION_ACCENT.get(s["num"], "#21262d")
        cards.append(
            f"<div style='background:#161b22; border:1px solid #21262d; border-radius:12px; "
            f"padding:14px; border-top:2px solid {accent};'>"
            f"<div style='color:{accent}; font-size:0.7rem; text-transform:uppercase; "
            f"letter-spacing:0.09em; margin-bottom:8px; font-weight:700;'>{s['num']}. {s['title']}</div>"
            f"{_bullets_to_html(s['content'])}</div>"
        )

    return header + "<div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>" + "".join(cards) + "</div>"


def analyze_ticker_action(ticker: str) -> tuple[str, str, str]:
    ticker = ticker.strip().upper()
    if not ticker:
        return "", "Enter a ticker symbol.", ""
    chart = tradingview_embed_html(ticker)
    try:
        text, stock_data = get_or_fetch_analysis(ticker)
        return chart, f"Analysis ready for **{ticker}**", render_analysis_html(ticker, text, stock_data)
    except Exception as exc:
        return chart, f"Failed: {exc}", f"<p style='color:#ef4444;'>Error: {exc}</p>"


# ── Batch watchlist analysis ──────────────────────────────────────────────────

def _section_text(text: str, num: str) -> str:
    m = re.search(rf"^{num}\.\s+.+$(.*?)(?=^\d+\.\s|\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


_REC_NORM = {
    "strong buy":  "Strong Buy",
    "strongbuy":   "Strong Buy",
    "strong_buy":  "Strong Buy",
    "strong sell": "Strong Sell",
    "strongsell":  "Strong Sell",
    "strong_sell": "Strong Sell",
    "buy":         "Buy",
    "hold":        "Hold",
    "sell":        "Sell",
}


def _parse_recommendation(text: str, stock_data: dict | None = None) -> str:
    # 1. yfinance analyst_summary is most reliable: "Consensus: strong_buy"
    if stock_data:
        m = re.search(r"Consensus:\s*([^\n]+)", stock_data.get("analyst_summary", ""), re.IGNORECASE)
        if m:
            raw = m.group(1).strip().lower().replace("_", "").replace(" ", "")
            for key, val in _REC_NORM.items():
                if key.replace("_", "").replace(" ", "") == raw:
                    return val

    # 2. Fallback: parse GPT section 4
    content = _section_text(text, "4").lower()
    for key, val in _REC_NORM.items():
        if key in content:
            return val
    return "—"


def _parse_confidence(text: str) -> str:
    # Strategy 1: find "11. Confidence …" header then grab first digit 1-10 after it
    m = re.search(r"11\.\s+Confidence[^\n]*\n(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        tail = m.group(1)
        d = re.search(r"\b(10|[1-9])\b", tail)
        if d:
            return d.group(1)
    # Strategy 2: any "N/10" pattern anywhere in the text
    m2 = re.search(r"\b(10|[1-9])\s*/\s*10\b", text)
    if m2:
        return m2.group(1)
    return "?"


def _news_highlight(text: str, stock_data: dict | None = None) -> str:
    # Prefer GPT section 2 (has real synthesized news even when yfinance is empty)
    gpt_news = _section_text(text, "2")
    for line in gpt_news.split("\n"):
        line = line.lstrip("-• ").strip()
        if len(line) > 10:
            return line[:100] + ("…" if len(line) > 100 else "")
    # Fallback: yfinance headlines
    raw = (stock_data or {}).get("news_summary", "")
    for line in raw.split("\n"):
        line = line.lstrip("- ").strip()
        if len(line) > 10 and "no recent" not in line.lower():
            return line[:100] + ("…" if len(line) > 100 else "")
    return "—"


def _analyze_one(ticker: str) -> None:
    try:
        get_or_fetch_analysis(ticker)
        with _batch_lock:
            _batch_status[ticker] = "done"
    except Exception as exc:
        with _batch_lock:
            _batch_status[ticker] = f"error: {exc}"


def start_batch_analysis(symbols: list[str]) -> str:
    with _batch_lock:
        _batch_status.clear()
        for s in symbols:
            _batch_status[s] = "queued"
    for sym in symbols:
        threading.Thread(target=_analyze_one, args=(sym,), daemon=True).start()
    return f"⚙ Analysis running in background for **{len(symbols)} symbols**. Click **Refresh Analysis Table** to see results as they complete."


def build_analysis_table(symbols: list[str]) -> str:
    if not symbols:
        return "<p style='color:#94a3b8; padding:16px;'>No symbols in watchlist.</p>"

    _REC_COLOR = {
        "Strong Buy": "#00d68f", "Buy": "#a3d68f",
        "Hold": "#d6c56f", "Sell": "#f87171", "Strong Sell": "#f04040",
    }

    rows = []
    for sym in sorted(symbols):
        with _batch_lock:
            status = _batch_status.get(sym)

        doc = analysis_collection.find_one({"ticker": sym})
        tv  = tradingview_url(sym)

        ticker_cell = (
            f"<a href='{tv}' target='_blank' "
            f"style='color:#00d68f; font-weight:700; text-decoration:none;'>{sym}</a>"
        )

        # Still queued / running in background
        if status == "queued" and not doc:
            rows.append(
                f"<tr><td>{ticker_cell}</td>"
                f"<td colspan=6 style='color:#4a5568; font-style:italic;'>Analyzing…</td></tr>"
            )
            continue

        if not doc:
            rows.append(
                f"<tr><td>{ticker_cell}</td>"
                f"<td colspan=6 style='color:#4a5568;'>Not analyzed yet</td></tr>"
            )
            continue

        sd     = doc.get("stock_data", {})
        text   = doc.get("analysis", "")

        # Price + change
        price  = sd.get("current_price")
        change = sd.get("daily_change") or ""
        chg_c  = "#00d68f" if "+" in change else "#f04040"
        price_cell = (
            f"<span style='color:#e2e8f4; font-weight:600;'>"
            f"{'${:,.2f}'.format(price) if price else 'N/A'}</span>"
            f"<br><span style='color:{chg_c}; font-size:0.78rem;'>{change}</span>"
        )

        # Analyst recommendation
        rec      = _parse_recommendation(text, sd)
        rec_c    = _REC_COLOR.get(rec, "#7a8499")
        rec_cell = (
            f"<span style='background:{rec_c}22; color:{rec_c}; border:1px solid {rec_c}44; "
            f"border-radius:6px; padding:3px 10px; font-size:0.8rem; font-weight:700;'>{rec}</span>"
        )

        # Analyst target + upside
        analyst    = sd.get("analyst_summary", "")
        target_cell = "—"
        tm = re.search(r"Mean Target:\s*\$?([\d.]+)", analyst)
        if tm:
            tval = float(tm.group(1))
            upside = ((tval - price) / price * 100) if price else None
            up_str = f" <span style='color:{'#00d68f' if (upside or 0)>=0 else '#f04040'}; font-size:0.78rem;'>({upside:+.1f}%)</span>" if upside is not None else ""
            target_cell = f"<span style='color:#e2e8f4;'>${tval:,.2f}</span>{up_str}"

        # News highlight
        first_news = _news_highlight(text, sd)

        # Confidence badge
        conf      = _parse_confidence(text)
        conf_val  = int(conf) if conf.isdigit() else 0
        conf_c    = "#00d68f" if conf_val >= 7 else "#d6c56f" if conf_val >= 5 else "#f04040"
        conf_cell = f"<span style='color:{conf_c}; font-weight:700; font-size:1rem;'>{conf}<span style='color:#4a5568; font-size:0.7rem;'>/10</span></span>"

        # Last updated
        updated   = doc.get("fetched_at")
        upd_str   = updated.strftime("%m/%d %H:%M") if updated else "?"

        rows.append(
            f"<tr>"
            f"<td>{ticker_cell}</td>"
            f"<td>{price_cell}</td>"
            f"<td>{rec_cell}</td>"
            f"<td>{target_cell}</td>"
            f"<td style='color:#7a8499; font-size:0.82rem;'>{first_news}</td>"
            f"<td style='text-align:center;'>{conf_cell}</td>"
            f"<td style='color:#4a5568; font-size:0.75rem; white-space:nowrap;'>{upd_str}</td>"
            f"</tr>"
        )

    analyzed = sum(1 for s in symbols if analysis_collection.find_one({"ticker": s}))
    header_note = (
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>"
        f"<span style='color:#4a5568; font-size:0.78rem;'>{analyzed} / {len(symbols)} analyzed · cached 5 days</span>"
        f"</div>"
    )

    table = (
        "<style>"
        "table.at{width:100%;border-collapse:collapse;font-size:0.85rem;}"
        "table.at th{background:#161b22;color:#4a5568;padding:9px 12px;border-bottom:1px solid #21262d;"
        "text-align:left;font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;}"
        "table.at td{background:#0d1117;color:#e2e8f4;padding:10px 12px;border-bottom:1px solid #161b22;"
        "vertical-align:middle;}"
        "table.at tr:hover td{background:#13181f;}"
        "</style>"
        "<table class='at'><thead><tr>"
        "<th>TICKER</th><th>PRICE</th><th>RECOMMENDATION</th>"
        "<th>ANALYST TARGET</th><th>NEWS HIGHLIGHT</th><th>CONF.</th><th>UPDATED</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return header_note + table


def batch_analyze_action(symbols: list[str]) -> tuple[str, str]:
    status = start_batch_analysis(symbols)
    return status, build_analysis_table(symbols)


def refresh_analysis_table_action(symbols: list[str]) -> str:
    return build_analysis_table(symbols)
