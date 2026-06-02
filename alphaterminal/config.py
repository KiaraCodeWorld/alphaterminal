import re

MONGO_URI         = "mongodb://localhost:27017/"
IBKR_HOST         = "127.0.0.1"
IBKR_PORT         = 7497

FASTMAIL_API_KEY  = "fmu1-38a643a9-a00c4f858ccc2b6fb2da15caccf59a15-0-320edeed9edf12d677eff045a7eeda81"
FASTMAIL_JMAP_URL = "https://api.fastmail.com/jmap/session"
FASTMAIL_HEADERS  = {
    "Authorization": f"Bearer {FASTMAIL_API_KEY}",
    "Content-Type":  "application/json",
}

ALERT_PATTERN = re.compile(
    r"Alert:\s*New symbols?:\s*([\w,\s]+?)\s+(?:was|were)\s+added\s+to\s+([\w]+)",
    re.IGNORECASE,
)

OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_WATCHLIST = [
    'WMT', 'MU', 'BMNR', 'PYPL', 'RBLX', 'SES', 'HIMS', 'ONDS', 'NFLX', 'SLDB',
    'COIN', 'MRNA', 'OKLO', 'FUBO', 'OSCR', 'TSM', 'V', 'META', 'BP', 'MSFT',
    'T', 'CXW', 'QSI', 'SNOW', 'DIS', 'NVDA', 'NBIS', 'BRK.B', 'CCJ', 'ARQT',
    'IBM', 'ONON', 'PG', 'TSLA', 'KO', 'ADBE', 'BABA', 'XYZ', 'AFRM', 'FLNA',
    'RGTI', 'XPEV', 'SHEL', 'UBER', 'MARA', 'NVO', 'SPY', 'JPM', 'MS', 'ANET',
    'SLV', 'GPRO', 'AAPL', 'BA', 'ORCL', 'AMZN', 'GLD', 'GOOG', 'NEM', 'BTC',
    'MSTR', 'IONQ', 'CCL', 'AAL', 'C', 'GRAB', 'IREN', 'AMD', 'DAL', 'UAL',
    'MRVL', 'NET', 'INTC',
]
