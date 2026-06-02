"""
Order execution layer.
Uses IBKR via ib_insync when connected; falls back to mock/paper mode.
"""
from datetime import datetime


def place_order(symbol: str, side: str = "BUY", quantity: int = 1) -> dict:
    """Submit a market order. Returns a result dict."""
    from .db import ib
    symbol = symbol.strip().upper()

    if ib.isConnected():
        try:
            from ib_insync import Stock, MarketOrder
            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            trade = ib.placeOrder(contract, MarketOrder(side, quantity))
            ib.sleep(0.5)
            return {
                "success":  True,
                "order_id": trade.order.orderId,
                "status":   trade.orderStatus.status,
                "broker":   "IBKR",
                "symbol":   symbol,
                "side":     side,
                "quantity": quantity,
                "placed_at": datetime.now().isoformat(),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "broker": "IBKR"}

    # Mock / paper mode — IBKR not connected
    print(f"[MOCK ORDER] {side} {quantity}x {symbol}")
    return {
        "success":  True,
        "order_id": None,
        "status":   "PaperFilled",
        "broker":   "mock",
        "symbol":   symbol,
        "side":     side,
        "quantity": quantity,
        "placed_at": datetime.now().isoformat(),
        "note":     "IBKR not connected — paper order only",
    }
