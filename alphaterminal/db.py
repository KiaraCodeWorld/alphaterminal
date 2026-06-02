import random
from pymongo import MongoClient
from ib_insync import IB
from .config import MONGO_URI, IBKR_HOST, IBKR_PORT

# ── MongoDB ──────────────────────────────────────────────────────────────────
_client = MongoClient(MONGO_URI)
_db     = _client["market_watchlist"]

watchlist_collection = _db["watchlist"]
alerts_collection    = _db["email_alerts"]
analysis_collection  = _db["stock_analysis"]
notes_collection           = _db["notes"]
trades_collection          = _db["trades_log"]
fire_portfolios_collection = _db["fire_portfolios"]
fire_holdings_collection   = _db["fire_holdings"]
crypto_collection          = _db["crypto_watchlist"]
new_finds_collection       = _db["new_finds"]

# ── IBKR ─────────────────────────────────────────────────────────────────────
ib = IB()
try:
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=random.randint(1000, 9999))
except Exception as exc:
    print(f"IBKR connection failed: {exc}")
