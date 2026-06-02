def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol in {"BRK.B", "BRK-B"}:
        return "BRK.B"
    return symbol


def tradingview_url(symbol: str) -> str:
    base = symbol.strip().upper().replace(".", "-")
    return f"https://www.tradingview.com/symbols/{base}/"
