import requests
from datetime import datetime, timedelta
from .config import FASTMAIL_HEADERS, FASTMAIL_JMAP_URL, ALERT_PATTERN
from .db import alerts_collection
from .utils import tradingview_url


def _add_weekdays(start: datetime, days: int) -> datetime:
    current, count = start, 0
    while count < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


def _fastmail_session() -> tuple[str | None, str | None]:
    try:
        r    = requests.get(FASTMAIL_JMAP_URL, headers=FASTMAIL_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return (
            data.get("apiUrl"),
            data.get("primaryAccounts", {}).get("urn:ietf:params:jmap:mail"),
        )
    except Exception as exc:
        print(f"⚠️  Fastmail session failed: {exc}")
        return None, None


def _fastmail_inbox_id(api_url: str, account_id: str) -> str | None:
    payload = {
        "using":       ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
        "methodCalls": [["Mailbox/get", {"accountId": account_id, "ids": None}, "0"]],
    }
    try:
        r = requests.post(api_url, json=payload, headers=FASTMAIL_HEADERS, timeout=10)
        r.raise_for_status()
        for mb in r.json()["methodResponses"][0][1]["list"]:
            if mb.get("role") == "inbox":
                return mb["id"]
    except Exception as exc:
        print(f"⚠️  Fastmail mailbox list failed: {exc}")
    return None


def parse_alert_subject(subject: str) -> dict | None:
    m = ALERT_PATTERN.search(subject)
    if not m:
        return None
    return {
        "symbols":   [s.strip().upper() for s in m.group(1).split(",") if s.strip()],
        "watchlist": m.group(2).strip(),
    }


def fetch_and_store_alerts() -> str:
    api_url, account_id = _fastmail_session()
    if not api_url:
        return "Failed to connect to Fastmail."

    inbox_id = _fastmail_inbox_id(api_url, account_id)
    if not inbox_id:
        return "Could not locate inbox."

    try:
        r = requests.post(api_url, headers=FASTMAIL_HEADERS, timeout=15, json={
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [["Email/query", {
                "accountId": account_id,
                "filter":    {"inMailbox": inbox_id, "subject": "Alert:"},
                "sort":      [{"property": "receivedAt", "isAscending": False}],
                "limit":     200,
            }, "0"]],
        })
        r.raise_for_status()
        email_ids = r.json()["methodResponses"][0][1].get("ids", [])
    except Exception as exc:
        return f"Email query failed: {exc}"

    if not email_ids:
        return "No alert emails found in inbox."

    try:
        r2 = requests.post(api_url, headers=FASTMAIL_HEADERS, timeout=15, json={
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [["Email/get", {
                "accountId":  account_id,
                "ids":        email_ids,
                "properties": ["id", "subject", "receivedAt"],
            }, "1"]],
        })
        r2.raise_for_status()
        emails = r2.json()["methodResponses"][0][1].get("list", [])
    except Exception as exc:
        return f"Email fetch failed: {exc}"

    new_count = 0
    for email in emails:
        parsed = parse_alert_subject(email.get("subject", ""))
        if not parsed:
            continue
        try:
            received_at = datetime.fromisoformat(
                email.get("receivedAt", "").replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            received_at = datetime.now()

        expires_at = _add_weekdays(received_at, 3)
        for symbol in parsed["symbols"]:
            if not alerts_collection.find_one({"email_id": email["id"], "symbol": symbol}):
                alerts_collection.insert_one({
                    "email_id":    email["id"],
                    "symbol":      symbol,
                    "watchlist":   parsed["watchlist"],
                    "received_at": received_at,
                    "expires_at":  expires_at,
                    "status":      "new",
                })
                new_count += 1

    return f"Checked {len(emails)} email(s), stored {new_count} new entry/entries."


def get_active_alerts() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for doc in alerts_collection.find({"expires_at": {"$gt": datetime.now()}}).sort("received_at", -1):
        grouped.setdefault(doc["watchlist"], []).append(doc)
    return grouped


def build_alerts_html() -> str:
    grouped = get_active_alerts()
    if not grouped:
        return "<p style='color:#94a3b8; padding:18px;'>No active alerts. Click <b>Refresh Alerts</b> to fetch from Fastmail.</p>"

    sections = []
    for watchlist, entries in sorted(grouped.items()):
        seen: dict[str, dict] = {}
        for e in entries:
            seen.setdefault(e["symbol"], e)

        chips = "".join(
            f"<a href='{tradingview_url(sym)}' target='_blank' style='text-decoration:none;'>"
            f"<span style='display:inline-flex; flex-direction:column; align-items:center; "
            f"background:#0d2b1a; color:#00d68f; border:1px solid #00d68f44; "
            f"border-radius:8px; padding:6px 14px; margin:4px; font-weight:700; font-size:0.85rem; "
            f"min-width:52px; text-align:center; line-height:1.3;'>"
            f"{sym}"
            f"<span style='color:#2a6644; font-size:0.65rem; font-weight:400; margin-top:1px;'>"
            f"EXP {seen[sym]['expires_at'].strftime('%m/%d')}</span></span></a>"
            for sym in sorted(seen)
        )
        sections.append(
            f"<div style='background:#161b22; border:1px solid #21262d; border-radius:12px; "
            f"padding:16px 18px; margin-bottom:12px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
            f"<span style='color:#e2e8f4; font-size:0.9rem; font-weight:600;'>{watchlist}</span>"
            f"<span style='color:#4a5568; font-size:0.75rem;'>{len(seen)} symbol{'s' if len(seen)!=1 else ''}</span>"
            f"</div><div style='display:flex; flex-wrap:wrap; gap:2px;'>{chips}</div></div>"
        )
    return "".join(sections)


def refresh_alerts_action() -> tuple[str, str]:
    return fetch_and_store_alerts(), build_alerts_html()
