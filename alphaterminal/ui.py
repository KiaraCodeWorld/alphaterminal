import gradio as gr
from .data import fetch_fear_greed_index
from .watchlist import (
    get_watchlist_symbols,
    refresh_dipbuy_dashboard,
    refresh_watchlist_html,
    add_symbol_action,
    remove_symbol_action,
)
from .alerts import build_alerts_html, refresh_alerts_action
from .analysis import (
    analyze_ticker_action,
    batch_analyze_action,
    refresh_analysis_table_action,
    build_analysis_table,
)
from .notes import add_note_action, delete_note_action, build_notes_html
from .trades import (
    add_trade_action, close_trade_action, delete_trade_action,
    refresh_prices_action, refresh_trades_table_action, save_edit_action,
    load_trade_for_edit, build_trades_html,
)

_SORT_SCRIPT = """
<script>
  window.deleteNote = function(noteId) {
    var inp = document.getElementById('delete-note-id-input');
    if (!inp) return;
    inp.value = noteId;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    var btn = document.getElementById('delete-note-btn');
    if (btn) btn.click();
  };

  window.deleteTrade = function(tradeId) {
    if (!confirm('Delete this trade? This cannot be undone.')) return;
    var inp = document.getElementById('trade-del-id');
    if (!inp) return;
    inp.value = tradeId;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    setTimeout(function() {
      var btn = document.getElementById('trade-del-btn');
      if (btn) btn.click();
    }, 30);
  };

  window.closeTrade = function(tradeId) {
    var ep = prompt('Enter exit price to close this trade:', '');
    if (ep === null || ep.trim() === '') return;
    ep = ep.trim();
    if (isNaN(parseFloat(ep))) { alert('Please enter a valid number.'); return; }
    var idInp = document.getElementById('trade-close-id');
    var prInp = document.getElementById('trade-close-price');
    if (!idInp || !prInp) return;
    idInp.value = tradeId;
    prInp.value = ep;
    idInp.dispatchEvent(new Event('input', {bubbles: true}));
    prInp.dispatchEvent(new Event('input', {bubbles: true}));
    setTimeout(function() {
      var btn = document.getElementById('trade-close-btn');
      if (btn) btn.click();
    }, 50);
  };

  window.editTrade = function(tradeId) {
    var inp = document.getElementById('trade-edit-id');
    if (!inp) return;
    inp.value = tradeId;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    setTimeout(function() {
      var btn = document.getElementById('trade-edit-load-btn');
      if (btn) btn.click();
    }, 30);
  };

  window.sortWatchlistTable = function(colIndex) {
    var table = document.querySelector('table.watchlist-table');
    if (!table) return;
    var tbody  = table.tBodies[0];
    var rows   = Array.from(tbody.rows);
    var prevCol = table.getAttribute('data-sort-col');
    var prevDir = table.getAttribute('data-sort-dir') || 'asc';
    var dir     = (prevCol === String(colIndex) && prevDir === 'asc') ? 'desc' : 'asc';

    rows.sort(function(a, b) {
      var av = a.cells[colIndex].getAttribute('data-sort') || a.cells[colIndex].textContent.trim();
      var bv = b.cells[colIndex].getAttribute('data-sort') || b.cells[colIndex].textContent.trim();
      var an = parseFloat(av.replace(/[^0-9.-]/g, ''));
      var bn = parseFloat(bv.replace(/[^0-9.-]/g, ''));
      if (!isNaN(an) && !isNaN(bn)) return dir === 'asc' ? an - bn : bn - an;
      return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
    table.setAttribute('data-sort-col', colIndex);
    table.setAttribute('data-sort-dir', dir);
    table.querySelectorAll('th.sortable').forEach(function(th, i) {
      th.classList.toggle('sort-active', i === colIndex);
      var arrow = th.querySelector('.sort-arrow');
      if (arrow) arrow.textContent = (i === colIndex) ? (dir === 'asc' ? '▲' : '▼') : '⇅';
    });
  };
</script>
"""

_CSS = """
    body, .gradio-container, .main { background:#0d1117 !important; color:#e2e8f4; }
    .gradio-container { max-width:100% !important; padding:0 !important; }
    footer { display:none !important; }
    .tab-nav button { background:transparent !important; color:#7a8499 !important;
        border:none !important; border-bottom:2px solid transparent !important;
        font-size:0.92rem !important; padding:10px 18px !important; border-radius:0 !important; }
    .tab-nav button.selected { color:#e2e8f4 !important;
        border-bottom:2px solid #e05a2e !important; font-weight:600 !important; }
    button.primary, .gr-button-primary { background:#e05a2e !important;
        color:#fff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
    button.secondary, .gr-button-secondary { background:#1e2536 !important;
        color:#e2e8f4 !important; border:1px solid #2a3347 !important; border-radius:8px !important; }
    button.primary:hover { background:#c44a22 !important; }
    input, textarea, .gr-input, .gr-textbox input { background:#161b22 !important;
        border:1px solid #21262d !important; color:#e2e8f4 !important; border-radius:8px !important; }
    .gr-markdown, .gr-markdown p { color:#7a8499 !important; }
"""

_HEADER_HTML = """
<div style='background:#161b22; border-bottom:1px solid #21262d; padding:12px 24px;
            display:flex; align-items:center; justify-content:space-between;'>
  <div>
    <span style='color:#e05a2e; font-size:1.35rem; font-weight:800; letter-spacing:-0.02em;'>AlphaTerminal</span>
    <div style='color:#4a5568; font-size:0.72rem; letter-spacing:0.08em; margin-top:1px;'>GLOBAL TERMINAL ACCESS</div>
  </div>
  <span style='background:#0d2b1a; color:#00d68f; font-size:0.75rem; font-weight:600;
               padding:4px 10px; border-radius:20px; border:1px solid #00d68f44;'>● SYSTEM LIVE</span>
</div>
"""


def _nav_html() -> str:
    nav_items = [
        ("Dashboard", "▣"), ("Dip Buy", "📈"), ("Watchlist", "☰"),
        ("Portfolio", "💼"), ("Reports", "📄"),
    ]
    rows = "".join(
        f"<div style='display:flex; align-items:center; gap:10px; padding:10px 12px; margin:2px 0; "
        f"border-radius:8px; background:{'#1e2a1e' if n=='Dip Buy' else 'transparent'}; "
        f"color:{'#00d68f' if n=='Dip Buy' else '#7a8499'};'>"
        f"<span>{ico}</span>"
        f"<span style='font-size:0.88rem; font-weight:{'600' if n=='Dip Buy' else '400'};'>{n}</span></div>"
        for n, ico in nav_items
    )
    return (
        f"<div style='background:#161b22; border:1px solid #21262d; border-radius:14px; "
        f"padding:20px 14px; min-height:520px; display:flex; flex-direction:column;'>"
        f"<div style='color:#4a5568; font-size:0.68rem; letter-spacing:0.1em; margin-bottom:16px;'>NAVIGATION</div>"
        f"{rows}"
        f"<div style='flex:1;'></div>"
        f"<div style='border-top:1px solid #21262d; padding-top:16px; margin-top:16px;'>"
        f"<div style='color:#4a5568; font-size:0.68rem; letter-spacing:0.1em; margin-bottom:12px;'>SYSTEM</div>"
        f"<div style='padding:8px 12px; color:#7a8499; font-size:0.88rem;'>⚙ Settings</div>"
        f"<div style='padding:8px 12px; color:#7a8499; font-size:0.88rem;'>❓ Support</div>"
        f"<div style='margin-top:14px; background:#e05a2e; color:#fff; text-align:center; "
        f"padding:10px; border-radius:8px; font-size:0.82rem; font-weight:700;'>⬤ Live Dashboard</div>"
        f"</div></div>"
    )


def create_app() -> gr.Blocks:
    initial_top   = "<div style='padding:18px; color:#94a3b8;'>Loading metrics...</div>"
    initial_fg    = f"<div style='padding:18px; background:#161b22; border-radius:14px; color:#e2e8f4; border:1px solid #21262d;'><p style='margin:0; font-size:0.8rem; color:#4a5568;'>FEAR &amp; GREED INDEX</p><h3 style='margin:8px 0 0;'>{fetch_fear_greed_index()}</h3></div>"
    initial_table = "<p style='color:#94a3b8; padding:18px;'>Click Refresh to load market data.</p>"

    with gr.Blocks(title="AlphaTerminal", css=_CSS) as app:
        gr.HTML(_HEADER_HTML)
        gr.HTML(_SORT_SCRIPT)

        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                gr.HTML(_nav_html())

            with gr.Column(scale=4):
                summary_out = gr.HTML(value=initial_top)
                fg_out      = gr.HTML(value=initial_fg)

                with gr.Tabs():
                    # ── Dip Buy ───────────────────────────────────────────
                    with gr.Tab("Dip Buy"):
                        with gr.Row():
                            search_in  = gr.Textbox(label="Filter tickers", placeholder="e.g. AAPL")
                            search_btn = gr.Button("Search")
                        dipbuy_out  = gr.HTML(value=initial_table)
                        refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
                        refresh_btn.click(refresh_dipbuy_dashboard, outputs=[summary_out, fg_out, dipbuy_out])
                        search_btn.click(refresh_dipbuy_dashboard, inputs=[search_in], outputs=[summary_out, fg_out, dipbuy_out])

                    # ── Watchlist ─────────────────────────────────────────
                    with gr.Tab("Watchlist"):
                        watchlist_out = gr.HTML(value=refresh_watchlist_html())
                        with gr.Row():
                            add_in  = gr.Textbox(label="Add ticker",    placeholder="AAPL")
                            add_btn = gr.Button("Add",    variant="primary")
                            rm_in   = gr.Textbox(label="Remove ticker", placeholder="AAPL")
                            rm_btn  = gr.Button("Remove", variant="secondary")
                        wl_msg = gr.Markdown("_")
                        add_btn.click(add_symbol_action,   inputs=[add_in], outputs=[wl_msg, watchlist_out])
                        rm_btn.click(remove_symbol_action, inputs=[rm_in],  outputs=[wl_msg, watchlist_out])

                        gr.HTML("<hr style='border-color:#21262d; margin:18px 0;'>")
                        gr.Markdown("#### AI Analysis — All Watchlist Symbols")
                        with gr.Row():
                            batch_btn   = gr.Button("⚙ Refresh AI Analysis (Background)", variant="primary")
                            refresh_tbl = gr.Button("↻ Refresh Analysis Table")
                        batch_status  = gr.Markdown("_")
                        analysis_tbl  = gr.HTML(value=build_analysis_table(get_watchlist_symbols()))

                        def _batch(symbols=None):
                            syms = get_watchlist_symbols()
                            return batch_analyze_action(syms)

                        def _refresh_tbl(symbols=None):
                            return refresh_analysis_table_action(get_watchlist_symbols())

                        batch_btn.click(_batch,       outputs=[batch_status, analysis_tbl])
                        refresh_tbl.click(_refresh_tbl, outputs=[analysis_tbl])

                    # ── Alerts ────────────────────────────────────────────
                    with gr.Tab("Alerts"):
                        alerts_out = gr.HTML(value=build_alerts_html())
                        with gr.Row():
                            refresh_alerts_btn = gr.Button("Refresh Alerts (Fetch from Fastmail)", variant="primary")
                            alerts_status      = gr.Markdown("_")
                        refresh_alerts_btn.click(refresh_alerts_action, outputs=[alerts_status, alerts_out])

                        gr.HTML("<hr style='border-color:#21262d; margin:18px 0;'>")
                        gr.Markdown("#### Analyze a Symbol")
                        with gr.Row():
                            analyze_in  = gr.Textbox(label="Ticker", placeholder="e.g. AAPL", scale=3)
                            analyze_btn = gr.Button("Analyze", variant="primary", scale=1)
                            clear_btn   = gr.Button("Clear",   scale=1)
                        analysis_status = gr.Markdown("_")
                        chart_embed     = gr.HTML(value="")
                        analysis_panel  = gr.HTML(value="")

                        analyze_btn.click(analyze_ticker_action, inputs=[analyze_in],
                                          outputs=[chart_embed, analysis_status, analysis_panel])
                        clear_btn.click(lambda: ("", "_", ""),
                                        outputs=[chart_embed, analysis_status, analysis_panel])

                    # ── Notes ─────────────────────────────────────────────
                    with gr.Tab("Notes"):
                        gr.Markdown("#### Add a Note")
                        with gr.Row():
                            note_ticker_in = gr.Textbox(label="Ticker (optional)", placeholder="e.g. AAPL", scale=1)
                            note_text_in   = gr.Textbox(label="Note", placeholder="Your trading note…", scale=4)
                            note_add_btn   = gr.Button("Add Note", variant="primary", scale=1)
                        note_status = gr.Markdown("_")

                        gr.HTML("<hr style='border-color:#21262d; margin:14px 0;'>")
                        notes_tbl = gr.HTML(value=build_notes_html())

                        # Hidden controls wired to delete buttons inside the HTML table
                        delete_id_in  = gr.Textbox(visible=False, elem_id="delete-note-id-input")
                        delete_btn    = gr.Button("_delete", visible=False, elem_id="delete-note-btn")

                        note_add_btn.click(
                            add_note_action,
                            inputs=[note_ticker_in, note_text_in],
                            outputs=[note_status, notes_tbl],
                        )
                        delete_btn.click(
                            delete_note_action,
                            inputs=[delete_id_in],
                            outputs=[note_status, notes_tbl],
                        )

                    # ── Trade Monitor ─────────────────────────────────────
                    with gr.Tab("Trade Monitor"):

                        # ── Log new trade form ────────────────────────────
                        with gr.Accordion("➕ Log New Trade", open=False):
                            with gr.Row():
                                tm_symbol   = gr.Textbox(label="Symbol",   placeholder="AAPL", scale=1)
                                tm_strategy = gr.Textbox(label="Strategy", placeholder="Supertrend Breakout", scale=2)
                                tm_side     = gr.Dropdown(["LONG", "SHORT"], label="Side",   value="LONG", scale=1)
                                tm_status   = gr.Dropdown(["OPEN", "CLOSED"], label="Status", value="OPEN", scale=1)
                            with gr.Row():
                                tm_entry  = gr.Textbox(label="Entry Price",  placeholder="150.00", scale=1)
                                tm_qty    = gr.Textbox(label="Quantity",     placeholder="100",    scale=1)
                                tm_target = gr.Textbox(label="Target Price", placeholder="165.00", scale=1)
                                tm_sl     = gr.Textbox(label="Stop Loss",    placeholder="145.00", scale=1)
                            with gr.Row():
                                tm_notes    = gr.Textbox(label="Notes", placeholder="Trade rationale…", scale=3)
                                tm_tags     = gr.Textbox(label="Tags (comma-sep)", placeholder="swing,earnings", scale=2)
                                tm_add_btn  = gr.Button("Log Trade", variant="primary", scale=1)
                            tm_add_status = gr.Markdown("_")

                        # ── Filter / sort bar ─────────────────────────────
                        with gr.Row():
                            tm_filter   = gr.Dropdown(["ALL", "OPEN", "CLOSED"], label="Filter", value="ALL", scale=1)
                            tm_sort     = gr.Dropdown(["entry_date", "pnl_pct", "symbol", "status"],
                                                      label="Sort By", value="entry_date", scale=1)
                            tm_ref_price_btn = gr.Button("⟳ Refresh Prices", variant="secondary", scale=1)
                            tm_ref_tbl_btn   = gr.Button("↻ Refresh Table", scale=1)
                        tm_action_status = gr.Markdown("_")

                        trades_tbl = gr.HTML(value=build_trades_html())

                        # ── Edit panel ────────────────────────────────────
                        gr.HTML("<hr style='border-color:#21262d; margin:16px 0;'>")
                        gr.Markdown("#### Edit Trade")
                        tm_edit_status = gr.Markdown(
                            "_Click **✎ Edit** on any row above to load a trade here._"
                        )
                        with gr.Row():
                            tm_edit_strategy = gr.Textbox(label="Strategy", scale=2)
                            tm_edit_notes    = gr.Textbox(label="Notes",    scale=3)
                            tm_edit_tags     = gr.Textbox(label="Tags (comma-sep)", scale=2)
                            tm_save_btn      = gr.Button("Save Changes", variant="primary", scale=1)

                        # ── Hidden action controls ────────────────────────
                        tm_del_id       = gr.Textbox(visible=False, elem_id="trade-del-id")
                        tm_del_btn      = gr.Button("_d",  visible=False, elem_id="trade-del-btn")
                        tm_close_id     = gr.Textbox(visible=False, elem_id="trade-close-id")
                        tm_close_price  = gr.Textbox(visible=False, elem_id="trade-close-price")
                        tm_close_btn    = gr.Button("_c",  visible=False, elem_id="trade-close-btn")
                        tm_edit_id      = gr.Textbox(visible=False, elem_id="trade-edit-id")
                        tm_edit_load_btn = gr.Button("_e", visible=False, elem_id="trade-edit-load-btn")

                        # ── Wiring ────────────────────────────────────────
                        tm_add_btn.click(
                            add_trade_action,
                            inputs=[tm_symbol, tm_strategy, tm_side, tm_status,
                                    tm_entry, tm_qty, tm_target, tm_sl,
                                    tm_notes, tm_tags, tm_filter, tm_sort],
                            outputs=[tm_add_status, trades_tbl],
                        )
                        tm_ref_price_btn.click(
                            refresh_prices_action,
                            inputs=[tm_filter, tm_sort],
                            outputs=[tm_action_status, trades_tbl],
                        )
                        tm_ref_tbl_btn.click(
                            refresh_trades_table_action,
                            inputs=[tm_filter, tm_sort],
                            outputs=[trades_tbl],
                        )
                        tm_filter.change(
                            refresh_trades_table_action,
                            inputs=[tm_filter, tm_sort],
                            outputs=[trades_tbl],
                        )
                        tm_sort.change(
                            refresh_trades_table_action,
                            inputs=[tm_filter, tm_sort],
                            outputs=[trades_tbl],
                        )
                        tm_del_btn.click(
                            delete_trade_action,
                            inputs=[tm_del_id, tm_filter, tm_sort],
                            outputs=[tm_action_status, trades_tbl],
                        )
                        tm_close_btn.click(
                            close_trade_action,
                            inputs=[tm_close_id, tm_close_price, tm_filter, tm_sort],
                            outputs=[tm_action_status, trades_tbl],
                        )
                        tm_edit_load_btn.click(
                            load_trade_for_edit,
                            inputs=[tm_edit_id],
                            outputs=[tm_edit_strategy, tm_edit_notes, tm_edit_tags, tm_edit_status],
                        )
                        tm_save_btn.click(
                            save_edit_action,
                            inputs=[tm_edit_id, tm_edit_strategy, tm_edit_notes,
                                    tm_edit_tags, tm_filter, tm_sort],
                            outputs=[tm_edit_status, trades_tbl],
                        )

    return app
