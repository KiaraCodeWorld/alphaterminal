import numpy as np
from datetime import datetime
from .config import DEFAULT_WATCHLIST
from .db import watchlist_collection
from .data import fetch_market_data, fetch_fear_greed_index
from .utils import normalize_symbol, tradingview_url


def normalize_db_watchlist() -> None:
    watchlist_collection.update_many({"symbol": "BRK-B"}, {"$set": {"symbol": "BRK.B"}})


def initialize_watchlist() -> None:
    if watchlist_collection.count_documents({}) == 0:
        watchlist_collection.insert_many([
            {"symbol": s.upper(), "created_at": datetime.now()} for s in DEFAULT_WATCHLIST
        ])


normalize_db_watchlist()
initialize_watchlist()


def get_watchlist_symbols() -> list[str]:
    return [doc["symbol"] for doc in watchlist_collection.find().sort("symbol", 1)]


def filter_watchlist_symbols(query: str) -> list[str]:
    query = query.strip().upper()
    if not query:
        return get_watchlist_symbols()
    symbols = get_watchlist_symbols()
    q_clean = query.replace(".", "").replace("-", "")
    return [s for s in symbols if query in s or q_clean in s.replace(".", "").replace("-", "")]


def add_watchlist_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return "Please enter a valid ticker symbol."
    norm = normalize_symbol(symbol)
    if watchlist_collection.find_one({"symbol": norm}):
        return f"{symbol} already exists in watchlist."
    watchlist_collection.insert_one({"symbol": norm, "created_at": datetime.now()})
    return f"Added {symbol} to watchlist."


def remove_watchlist_symbol(symbol: str) -> str:
    norm   = normalize_symbol(symbol.strip().upper())
    result = watchlist_collection.delete_one({"symbol": norm})
    return f"Removed {symbol}." if result.deleted_count else f"{symbol} not found."


# ── HTML builders ─────────────────────────────────────────────────────────────

def build_dipbuy_table(symbols: list[str]) -> str:
    if not symbols:
        return "<p style='color:#94a3b8;'>No symbols matched your search.</p>"

    market_data = fetch_market_data(symbols, normalize_symbol)
    symbols = sorted(
        symbols,
        key=lambda s: market_data.get(s, {}).get("rsi", 9999)
                      if not market_data.get(s, {}).get("error") else 9999,
    )

    rows = []
    for symbol in symbols:
        data = market_data.get(symbol, {})
        if data.get("error"):
            rows.append(f"<tr><td>{symbol}</td><td colspan=8 style='color:#d9534f;'>Data unavailable</td></tr>")
            continue

        rsi = data["rsi"]
        if rsi < 30:
            rb, rf = "#0d2b1a", "#00d68f"
        elif rsi < 50:
            rb, rf = "#1a200d", "#a3d68f"
        elif rsi < 70:
            rb, rf = "#1a1a0d", "#d6c56f"
        else:
            rb, rf = "#2b0d0d", "#f04040"

        rsi_badge = (
            f"<span style='background:{rb}; color:{rf}; font-weight:700; font-size:0.82rem; "
            f"padding:3px 9px; border-radius:20px; border:1px solid {rf}44;'>{rsi:.1f}</span>"
        )
        is_below  = data["below_volume_profile"]
        dot_color = "#f04040" if is_below else "#00d68f"
        vol_badge = f"<span style='color:{dot_color}; font-size:0.82rem;'>● {'Below' if is_below else 'Above'}</span>"

        cc1  = "#00d68f" if data["change_pct"] >= 0 else "#f04040"
        cc7  = "#00d68f" if data["change_7d"]  >= 0 else "#f04040"
        cc30 = "#00d68f" if data["change_30d"] >= 0 else "#f04040"

        rows.append(
            "<tr>"
            f"<td><a href='{data['tradingview_url']}' target='_blank' "
            f"style='color:#00d68f; font-weight:600; text-decoration:none;'>{symbol}</a></td>"
            f"<td style='color:#e2e8f4; font-weight:600;'>${data['price']:,.2f}</td>"
            f"<td style='color:{cc1};'>{data['change_pct']:+.2f}%</td>"
            f"<td style='color:{cc7};'>{data['change_7d']:+.2f}%</td>"
            f"<td style='color:{cc30};'>{data['change_30d']:+.2f}%</td>"
            f"<td data-sort='{rsi:.3f}'>{rsi_badge}</td>"
            f"<td data-sort='{0 if is_below else 1}'>{vol_badge}</td>"
            f"<td style='color:#7a8499;'>${data['support']:,.2f}</td>"
            f"<td style='color:#7a8499;'>${data['resistance']:,.2f}</td>"
            "</tr>"
        )

    return (
        """<style>
        table.watchlist-table {width:100%; border-collapse:collapse; font-size:0.88rem;}
        table.watchlist-table th {
            background:#161b22; color:#4a5568; padding:10px 12px;
            border-bottom:1px solid #21262d; text-align:left; user-select:none;
            font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
        }
        table.watchlist-table td {
            background:#0d1117; color:#e2e8f4; padding:11px 12px;
            border-bottom:1px solid #161b22; text-align:left;
        }
        table.watchlist-table tr:hover td { background:#13181f; }
        table.watchlist-table th.sortable { cursor:pointer; white-space:nowrap; }
        table.watchlist-table th.sortable:hover { color:#e2e8f4; }
        table.watchlist-table th.sort-active { color:#e05a2e !important; }
        table.watchlist-table th .sort-arrow {
            display:inline-block; margin-left:5px; font-size:0.65rem;
            opacity:0.35; vertical-align:middle;
        }
        table.watchlist-table th.sort-active .sort-arrow { opacity:1; }
        </style>
        <table class='watchlist-table' data-sort-col='5' data-sort-dir='asc'>
          <thead><tr>
            <th class='sortable' onclick='sortWatchlistTable(0)'>TICKER<span class='sort-arrow'>⇅</span></th>
            <th class='sortable' onclick='sortWatchlistTable(1)'>PRICE<span class='sort-arrow'>⇅</span></th>
            <th class='sortable' onclick='sortWatchlistTable(2)'>TODAY %<span class='sort-arrow'>⇅</span></th>
            <th class='sortable' onclick='sortWatchlistTable(3)'>7D %<span class='sort-arrow'>⇅</span></th>
            <th class='sortable' onclick='sortWatchlistTable(4)'>30D %<span class='sort-arrow'>⇅</span></th>
            <th class='sortable sort-active' onclick='sortWatchlistTable(5)'>RSI<span class='sort-arrow'>▲</span></th>
            <th class='sortable' onclick='sortWatchlistTable(6)'>VOL PROFILE<span class='sort-arrow'>⇅</span></th>
            <th class='sortable' onclick='sortWatchlistTable(7)'>SUPPORT<span class='sort-arrow'>⇅</span></th>
            <th class='sortable' onclick='sortWatchlistTable(8)'>RESISTANCE<span class='sort-arrow'>⇅</span></th>
          </tr></thead>
          <tbody>"""
        + "".join(rows)
        + "</tbody></table>"
    )


def build_watchlist_html(symbols: list[str]) -> str:
    items = "".join(f"<li>{s}</li>" for s in symbols)
    return f"<div><strong>Watchlist ({len(symbols)} symbols)</strong><ul>{items}</ul></div>"


def build_top_cards(symbols: list[str]) -> str:
    market_data = fetch_market_data(symbols, normalize_symbol)
    valid       = [d for d in market_data.values() if not d.get("error")]
    oversold    = sum(1 for d in valid if d["rsi"] < 35)
    avg_change  = float(np.mean([d["change_pct"] for d in valid])) if valid else 0.0
    avg_color   = "#00d68f" if avg_change >= 0 else "#f04040"
    fg          = fetch_fear_greed_index()

    def _card(label, value, sub="", accent="#e2e8f4"):
        sub_html = f"<div style='color:#4a5568; font-size:0.78rem; margin-top:5px;'>{sub}</div>" if sub else ""
        return (
            f"<div style='background:#161b22; border:1px solid #21262d; border-radius:14px; padding:18px 20px;'>"
            f"<div style='color:#4a5568; font-size:0.68rem; font-weight:600; letter-spacing:0.1em; "
            f"text-transform:uppercase; margin-bottom:10px;'>{label}</div>"
            f"<div style='color:{accent}; font-size:1.55rem; font-weight:700; line-height:1.15;'>{value}</div>"
            f"{sub_html}</div>"
        )

    return (
        "<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px;'>"
        + _card("Fear &amp; Greed", fg)
        + _card("Watchlist Size", str(len(symbols)), f"{len(symbols)} symbols tracked")
        + _card("Oversold RSI", str(oversold),
                "Strong Buy Signal" if oversold > 5 else "Symbols below RSI 30",
                "#00d68f" if oversold > 0 else "#e2e8f4")
        + _card("Avg Daily Change", f"{avg_change:+.2f}%", f"Across {len(valid)} symbols", avg_color)
        + "</div>"
    )


# ── Gradio action handlers ────────────────────────────────────────────────────

def refresh_dipbuy_dashboard(filter_query: str = "") -> tuple[str, str, str]:
    symbols    = filter_watchlist_symbols(filter_query)
    top_html   = build_top_cards(symbols)
    fg_html    = (
        f"<div style='padding:18px; background:#161b22; border-radius:14px; color:#e2e8f4; border:1px solid #21262d;'>"
        f"<p style='margin:0; font-size:0.8rem; color:#4a5568;'>FEAR &amp; GREED INDEX</p>"
        f"<h3 style='margin:8px 0 0; font-size:1.3rem;'>{fetch_fear_greed_index()}</h3></div>"
    )
    table_html = build_dipbuy_table(symbols)
    return top_html, fg_html, table_html


def refresh_watchlist_html() -> str:
    return build_watchlist_html(get_watchlist_symbols())


def add_symbol_action(symbol: str) -> tuple[str, str]:
    return add_watchlist_symbol(symbol), refresh_watchlist_html()


def remove_symbol_action(symbol: str) -> tuple[str, str]:
    return remove_watchlist_symbol(symbol), refresh_watchlist_html()
