from datetime import datetime
from .db import notes_collection


def add_note(ticker: str, note_text: str) -> str:
    ticker = ticker.strip().upper()
    note_text = note_text.strip()
    if not note_text:
        return "Note text cannot be empty."
    notes_collection.insert_one({
        "ticker":     ticker or "GENERAL",
        "note":       note_text,
        "created_at": datetime.now(),
    })
    return f"Note saved{f' for {ticker}' if ticker else ''}."


def delete_note(note_id: str) -> str:
    from bson import ObjectId
    try:
        result = notes_collection.delete_one({"_id": ObjectId(note_id)})
        return "Deleted." if result.deleted_count else "Note not found."
    except Exception as exc:
        return f"Error: {exc}"


def get_all_notes() -> list[dict]:
    return list(notes_collection.find().sort("created_at", -1))


def build_notes_html() -> str:
    notes = get_all_notes()
    if not notes:
        return "<p style='color:#94a3b8; padding:18px;'>No notes yet. Add one above.</p>"

    rows = "".join(
        f"<tr>"
        f"<td style='color:#4a5568; font-size:0.75rem; white-space:nowrap;'>"
        f"{doc['created_at'].strftime('%m/%d/%y %H:%M')}</td>"
        f"<td style='color:#00d68f; font-weight:700; white-space:nowrap;'>{doc.get('ticker', '—')}</td>"
        f"<td style='color:#e2e8f4;'>{doc.get('note', '')}</td>"
        f"<td style='text-align:center;'>"
        f"<button onclick=\"deleteNote('{doc['_id']}')\" "
        f"style='background:#2b0d0d; color:#f04040; border:1px solid #f0404044; "
        f"border-radius:6px; padding:3px 10px; font-size:0.75rem; cursor:pointer; "
        f"font-weight:600;'>✕ Delete</button></td>"
        f"</tr>"
        for doc in notes
    )

    return (
        "<style>"
        "table.notes-table{width:100%;border-collapse:collapse;font-size:0.85rem;}"
        "table.notes-table th{background:#161b22;color:#4a5568;padding:9px 12px;"
        "border-bottom:1px solid #21262d;text-align:left;font-size:0.7rem;"
        "font-weight:700;letter-spacing:0.08em;text-transform:uppercase;}"
        "table.notes-table td{background:#0d1117;color:#e2e8f4;padding:10px 12px;"
        "border-bottom:1px solid #161b22;vertical-align:middle;}"
        "table.notes-table tr:hover td{background:#13181f;}"
        "</style>"
        "<table class='notes-table'><thead><tr>"
        "<th>DATE</th><th>TICKER</th><th>NOTE</th><th>ACTION</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


# ── Gradio action handlers ────────────────────────────────────────────────────

def add_note_action(ticker: str, note_text: str) -> tuple[str, str]:
    msg = add_note(ticker, note_text)
    return msg, build_notes_html()


def delete_note_action(note_id: str) -> tuple[str, str]:
    msg = delete_note(note_id.strip())
    return msg, build_notes_html()
