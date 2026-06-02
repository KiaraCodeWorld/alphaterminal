"""
AlphaTerminal — AI Strategy Lab
Autonomous multi-agent platform for researching, generating, validating,
stress-testing, and evolving stock trading strategies.
"""
import os
import re
import json
import uuid
import random
import textwrap
import threading
import traceback
import urllib.parse
from datetime import datetime, timedelta

import requests as _requests

import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from openai import OpenAI
from ib_insync import IB, Stock, util

from .config import IBKR_HOST, IBKR_PORT

# ═══════════════════════════════════════════════════════════════════════════════
# Job store
# ═══════════════════════════════════════════════════════════════════════════════

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _new_job(mode: str = "validate") -> str:
    jid = str(uuid.uuid4())
    with _lock:
        _jobs[jid] = {
            "status": "pending", "mode": mode,
            "step": "", "progress": 0,
            "log": [], "result": None, "error": None,
            "created_at": datetime.now().isoformat(),
            "tokens": {
                "haiku":    {"in": 0, "out": 0},
                "grok":     {"in": 0, "out": 0},
            },
        }
    return jid


def _upd(jid: str, **kw) -> None:
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return
        for k, v in kw.items():
            if k == "log+":
                job["log"].append(v)
            elif k == "tok+":
                # v = {"model": "haiku"|"grok", "in": N, "out": N}
                m = v.get("model", "grok")
                job["tokens"].setdefault(m, {"in": 0, "out": 0})
                job["tokens"][m]["in"]  += v.get("in", 0)
                job["tokens"][m]["out"] += v.get("out", 0)
            else:
                job[k] = v


def get_job(jid: str) -> dict | None:
    with _lock:
        return dict(_jobs.get(jid, {}))


# ═══════════════════════════════════════════════════════════════════════════════
# LLM clients & routing
# Grok   → all reasoning tasks (research, parse, agents, report)
# Haiku  → code generation only (generate_strategy_code, evolve signal_generator)
# ═══════════════════════════════════════════════════════════════════════════════

# Cost per 1M tokens (USD)
# DeepSeek-V3: $0.27 input / $1.10 output
# Claude Haiku 4.5: $1.00 input / $5.00 output
# Claude Sonnet 4.6: $3.00 input / $15.00 output
_COSTS = {
    "haiku":  {"in": 1.00,  "out": 5.00},
    "sonnet": {"in": 3.00,  "out": 15.00},
    "grok":   {"in": 0.27,  "out": 1.10},   # "grok" key kept for internal consistency
}

def _claude_model_key() -> str:
    """Returns 'sonnet' or 'haiku' for cost tracking based on active model."""
    return "sonnet" if "sonnet" in _active_claude_model["model"] else "haiku"
# Token warning threshold (total estimated cost per job, USD)
_COST_WARN_USD = 0.50

_HAIKU_MODEL   = "claude-haiku-4-5-20251001"
_SONNET_MODEL  = "claude-sonnet-4-6"
_DS_MODEL      = "deepseek-chat"
_DS_BASE       = "https://api.deepseek.com"

# Active Claude model for code generation — set per-job via start_lab_job
_active_claude_model: dict = {"model": _HAIKU_MODEL}


def _haiku_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _ds_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url=_DS_BASE)


def _grok(system: str, user: str, max_tokens: int = 1500, jid: str | None = None) -> str:
    """Call DeepSeek for reasoning / analysis tasks."""
    resp = _ds_client().chat.completions.create(
        model=_DS_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    if jid:
        u = resp.usage
        _upd(jid, **{"tok+": {"model": "grok",
                               "in": getattr(u, "prompt_tokens", 0),
                               "out": getattr(u, "completion_tokens", 0)}})
        _warn_tokens(jid)
    return resp.choices[0].message.content.strip()


def _haiku(system: str, user: str, max_tokens: int = 2000, jid: str | None = None) -> str:
    """Call Claude (Haiku or Sonnet) for code-generation tasks only."""
    model = _active_claude_model["model"]
    msg = _haiku_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if jid:
        u = msg.usage
        _upd(jid, **{"tok+": {"model": _claude_model_key(),
                               "in": getattr(u, "input_tokens", 0),
                               "out": getattr(u, "output_tokens", 0)}})
        _warn_tokens(jid)
    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    return text.strip()


def _token_cost(jid: str) -> dict:
    """Return per-model token counts and estimated costs."""
    with _lock:
        toks = _jobs.get(jid, {}).get("tokens", {})
    result = {}
    total_cost = 0.0
    for model, counts in toks.items():
        inp, out = counts.get("in", 0), counts.get("out", 0)
        c = _COSTS.get(model, {"in": 0, "out": 0})
        cost = inp / 1_000_000 * c["in"] + out / 1_000_000 * c["out"]
        total_cost += cost
        result[model] = {"in": inp, "out": out, "cost_usd": round(cost, 4)}
    result["total_cost_usd"] = round(total_cost, 4)
    return result


def _warn_tokens(jid: str) -> None:
    costs = _token_cost(jid)
    total = costs.get("total_cost_usd", 0)
    if total >= _COST_WARN_USD:
        _upd(jid, **{"log+": f"[TOKEN WARNING] Est. cost so far: ${total:.4f}"})


def _extract_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in response: {raw[:400]}")
    return json.loads(m.group())


# ═══════════════════════════════════════════════════════════════════════════════
# Interval helpers
# ═══════════════════════════════════════════════════════════════════════════════

_INTERVAL_LABEL = {
    "1m": "1-minute", "2m": "2-minute", "5m": "5-minute",
    "15m": "15-minute", "30m": "30-minute", "1h": "hourly",
    "90m": "90-minute", "1d": "daily", "1wk": "weekly",
    "1mo": "monthly", "3mo": "quarterly",
}
_IBKR_BAR = {
    "1m": "1 min", "2m": "2 mins", "5m": "5 mins",
    "15m": "15 mins", "30m": "30 mins", "1h": "1 hour",
    "90m": "90 mins", "1d": "1 day", "1wk": "1 week",
    "1mo": "1 month", "3mo": "1 month",
}


def _norm_interval(raw: str | None) -> str:
    if not raw:
        return "1d"
    r = raw.strip().lower()
    m = {
        "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
        "60m": "1h", "1h": "1h", "90m": "90m", "1d": "1d", "daily": "1d",
        "5d": "5d", "1wk": "1wk", "weekly": "1wk",
        "1mo": "1mo", "monthly": "1mo", "3mo": "3mo", "quarterly": "3mo",
    }
    if r in m:
        return m[r]
    if "week" in r:  return "1wk"
    if "month" in r: return "1mo"
    if "hour" in r:  return "1h"
    if "30" in r and "min" in r: return "30m"
    if "15" in r and "min" in r: return "15m"
    if "5"  in r and "min" in r: return "5m"
    if "intraday" in r: return "1h"
    return "1d"


def _valid_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# IBKR + yfinance data fetching
# ═══════════════════════════════════════════════════════════════════════════════

def _ibkr_duration(start: str, end: str) -> str:
    try:
        days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    except Exception:
        return "3 Y"
    if days <= 7:   return f"{days} D"
    if days <= 30:  return f"{max(1, days // 7)} W"
    months = days // 30
    if months <= 11: return f"{months} M"
    return f"{max(1, days // 365)} Y"


def _fetch_ibkr(ib: IB, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    bar_size = _IBKR_BAR.get(interval, "1 day")
    duration = _ibkr_duration(start, end)
    try:
        end_dt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d 23:59:59")
    except Exception:
        end_dt = ""
    try:
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime=end_dt, durationStr=duration,
            barSizeSetting=bar_size, whatToShow="TRADES",
            useRTH=True, formatDate=1,
        )
        if not bars:
            return None
        df = util.df(bars)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                  "close": "Close", "volume": "Volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df if not df.empty else None
    except Exception:
        return None


def _fetch_yfinance(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol.replace(".", "-"), start=start, end=end,
            interval=interval, auto_adjust=True, progress=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _fetch_bars(symbol: str, start: str, end: str, interval: str,
                ib: IB | None = None) -> pd.DataFrame:
    if ib and ib.isConnected():
        df = _fetch_ibkr(ib, symbol, start, end, interval)
        if df is not None and len(df) >= 10:
            return df
    df = _fetch_yfinance(symbol, start, end, interval)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} ({start}→{end}, {interval})")
    return df


def _connect_ibkr() -> IB:
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=random.randint(2000, 8999))
    except Exception:
        pass
    return ib


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest engine
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    def __init__(self, symbols: list[str], start: str, end: str,
                 capital: float = 100_000, slippage: float = 0.0005,
                 commission: float = 0.005, stop_loss: float | None = None,
                 interval: str = "1d"):
        self.symbols   = symbols
        self.start     = start
        self.end       = end
        self.capital   = capital
        self.slippage  = slippage
        self.commission = commission
        self.stop_loss = stop_loss
        self.interval  = _norm_interval(interval)

    def run(self, signal_fn) -> dict:
        all_equity, all_trades, errors = [], [], []
        ib = _connect_ibkr()

        for symbol in self.symbols:
            try:
                df = _fetch_bars(symbol, self.start, self.end, self.interval,
                                 ib if ib.isConnected() else None)
                if len(df) < 10:
                    errors.append(f"{symbol}: {len(df)} bars"); continue

                # Pass lowercase-column copy so generated code works regardless of case
                df_sig = df.copy()
                df_sig.columns = [c.lower() for c in df_sig.columns]
                signals = signal_fn(df_sig)
                if not isinstance(signals, pd.Series):
                    signals = pd.Series(signals, index=df.index)
                signals = signals.reindex(df.index).fillna(0)

                cash = self.capital / len(self.symbols)
                pos, entry_price, entry_date = 0, 0.0, None
                equity_series = []

                for i in range(1, len(df)):
                    row, sig = df.iloc[i], signals.iloc[i - 1]

                    if sig == 1 and pos == 0:
                        fill = row["Open"] * (1 + self.slippage)
                        shares = int(cash // fill)
                        if shares > 0:
                            cash -= fill * shares + self.commission * shares
                            pos, entry_price, entry_date = shares, fill, df.index[i]

                    elif pos > 0:
                        stop_hit = (
                            self.stop_loss is not None
                            and row["Low"] <= entry_price * (1 - self.stop_loss)
                        )
                        if sig == -1 or stop_hit:
                            fill = (
                                entry_price * (1 - self.stop_loss)
                                if stop_hit else row["Open"] * (1 - self.slippage)
                            )
                            proceeds = fill * pos - self.commission * pos
                            pnl = proceeds - entry_price * pos
                            all_trades.append({
                                "symbol": symbol, "entry_date": entry_date,
                                "exit_date": df.index[i], "entry": entry_price,
                                "exit": fill, "shares": pos, "pnl": pnl,
                                "pnl_pct": pnl / (entry_price * pos) * 100,
                            })
                            cash += proceeds
                            pos = 0

                    equity_series.append({"date": df.index[i],
                                          "equity": cash + pos * row["Close"]})

                if equity_series:
                    all_equity.append(
                        pd.DataFrame(equity_series).set_index("date")["equity"]
                    )
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")

        try: ib.disconnect()
        except Exception: pass

        if not all_equity:
            detail = "; ".join(errors) if errors else "all symbols returned empty data"
            return {"error": f"No equity data — {detail}"}

        combined = pd.concat(all_equity, axis=1).sum(axis=1).sort_index()
        return {
            "equity": combined,
            "trades": pd.DataFrame(all_trades) if all_trades else pd.DataFrame(),
            "fetch_errors": errors,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity: pd.Series, trades: pd.DataFrame, capital: float) -> dict:
    if equity.empty:
        return {}
    rets  = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
    total_ret = (equity.iloc[-1] - capital) / capital * 100
    cagr      = ((equity.iloc[-1] / capital) ** (1 / years) - 1) * 100
    rf        = 0.04 / 252
    excess    = rets - rf
    sharpe    = (excess.mean() / rets.std() * 252**0.5) if rets.std() > 0 else 0
    down      = rets[rets < 0]
    sortino   = (excess.mean() / down.std() * 252**0.5) if len(down) > 1 else 0
    roll_max  = equity.cummax()
    dd        = (equity - roll_max) / roll_max * 100
    max_dd    = dd.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0

    wr = pf = exp = aw = al = 0.0
    if not trades.empty and "pnl" in trades.columns:
        wins = trades[trades["pnl"] > 0]["pnl"]
        loss = trades[trades["pnl"] <= 0]["pnl"]
        wr   = len(wins) / len(trades) * 100
        gp   = wins.sum(); gl = abs(loss.sum())
        pf   = gp / gl if gl > 0 else 999
        aw   = wins.mean() if len(wins) else 0
        al   = loss.mean() if len(loss) else 0
        exp  = wr / 100 * aw + (1 - wr / 100) * al

    def _f(v, d=2): return round(float(v), d)
    return {
        "total_return_pct": _f(total_ret),
        "cagr_pct":         _f(cagr),
        "sharpe":           _f(sharpe, 3),
        "sortino":          _f(sortino, 3),
        "max_drawdown_pct": _f(max_dd),
        "calmar":           _f(calmar, 3),
        "win_rate_pct":     _f(wr),
        "profit_factor":    _f(pf, 3),
        "expectancy_usd":   _f(exp),
        "total_trades":     int(len(trades)) if not trades.empty else 0,
        "avg_win_usd":      _f(aw),
        "avg_loss_usd":     _f(al),
        "final_equity":     _f(equity.iloc[-1]),
        "years":            _f(years),
    }


def _equity_points(equity: pd.Series, max_pts: int = 500) -> list[dict]:
    step = max(1, len(equity) // max_pts)
    return [{"date": str(idx.date()), "value": round(float(v), 2)}
            for idx, v in equity.iloc[::step].items()]


# ═══════════════════════════════════════════════════════════════════════════════
# Regime analysis
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regimes(start: str, end: str) -> dict:
    try:
        ib  = _connect_ibkr()
        df  = _fetch_bars("SPY", start, end, "1d", ib if ib.isConnected() else None)
        try: ib.disconnect()
        except Exception: pass
        close = df["Close"].dropna()
        rets  = close.pct_change().dropna()
        ma50  = close.rolling(50).mean()
        vol20 = rets.rolling(20).std() * 252**0.5
        bull  = (close > ma50).sum()
        bear  = (close < ma50).sum()
        total = len(close)
        return {
            "total_days":       int(total),
            "bull_pct":         round(float(bull / total * 100), 1),
            "bear_pct":         round(float(bear / total * 100), 1),
            "annualized_vol":   round(float(rets.std() * 252**0.5) * 100, 2),
            "high_vol_days":    int((vol20 > 0.20).sum()),
            "spy_total_return": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
        }
    except Exception:
        return {}


def regime_breakdown(trades: pd.DataFrame, start: str, end: str) -> dict:
    """Win rate / avg PnL per market regime based on trade entry dates."""
    if trades.empty or "entry_date" not in trades.columns:
        return {}
    try:
        ib  = _connect_ibkr()
        spy = _fetch_bars("SPY", start, end, "1d", ib if ib.isConnected() else None)
        try: ib.disconnect()
        except Exception: pass
        close = spy["Close"].dropna()
        ma50  = close.rolling(50).mean().dropna()
        rets  = close.pct_change().dropna()
        vol20 = rets.rolling(20).std() * 252**0.5

        def _label(dt):
            dt = pd.Timestamp(dt).tz_localize(None)
            if dt not in ma50.index:
                dt = ma50.index[ma50.index.get_indexer([dt], method="nearest")[0]]
            is_bull  = close.get(dt, 0) > ma50.get(dt, 0)
            is_hivol = vol20.get(dt, 0) > 0.20
            if is_hivol: return "High Vol"
            return "Bull" if is_bull else "Bear"

        trades = trades.copy()
        trades["regime"] = trades["entry_date"].apply(_label)
        result = {}
        for reg, grp in trades.groupby("regime"):
            wins = grp[grp["pnl"] > 0]
            result[reg] = {
                "trades":   len(grp),
                "win_rate": round(float(len(wins) / len(grp) * 100), 1),
                "avg_pnl":  round(float(grp["pnl"].mean()), 2),
            }
        return result
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Signal function builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_signal_fn(code: str):
    if not code or not code.strip():
        raise RuntimeError("Empty signal generator code")
    ns: dict = {"pd": pd, "np": np}
    try:
        import pandas_ta as _pta
        ns["ta"] = _pta
    except ImportError:
        pass
    try:
        import talib as _talib
        ns["talib"] = _talib
    except ImportError:
        pass
    try:
        exec(textwrap.dedent(code), ns)  # nosec
    except Exception as exc:
        raise RuntimeError(f"Compile error: {exc}") from exc
    fn = ns.get("generate_signals")
    if fn is None:
        raise RuntimeError("generate_signals(df) not found in extracted code")
    return fn


_FIX_SYSTEM = """\
You are a Python expert fixing a broken generate_signals(df) trading strategy function.
The DataFrame passed to the function has LOWERCASE columns: open, high, low, close, volume.
Fix ALL errors reported and return ONLY the corrected Python function — no markdown, no explanation.
Rules:
- Use lowercase column names: df['close'], df['open'], df['high'], df['low'], df['volume']
- Never use df['Close'], df['Open'] etc.
- Return a pd.Series with the same index as df, values 1 (buy) / -1 (sell) / 0 (hold)
- Handle NaN values with .fillna(0) or .dropna() appropriately
- Do not use any external libraries beyond pandas and numpy"""


def _autofix_signal_code(code: str, error: str, attempt: int, jid: str | None = None) -> str:
    """Ask Haiku to fix broken signal code. Returns corrected code string."""
    prompt = (
        f"Fix this generate_signals(df) function. Attempt {attempt}/3.\n\n"
        f"ERROR:\n{error}\n\n"
        f"BROKEN CODE:\n{code}\n\n"
        "Return ONLY the fixed Python function."
    )
    fixed = _haiku(_FIX_SYSTEM, prompt, max_tokens=1500, jid=jid)
    # Strip markdown fences if Haiku added them
    fixed = re.sub(r"^```python\s*", "", fixed.strip(), flags=re.IGNORECASE)
    fixed = re.sub(r"\s*```$", "", fixed.strip())
    return fixed.strip()


def _build_signal_fn_with_fix(code: str, jid: str | None = None, max_attempts: int = 3):
    """Compile signal function with automatic Haiku-powered fix loop."""
    current_code = code
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            fn = _build_signal_fn(current_code)
            if attempt > 1 and jid:
                _upd(jid, **{"log+": f"[Haiku] Code fixed on attempt {attempt}"})
            return fn, current_code
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_attempts:
                if jid:
                    _upd(jid, **{"log+": f"[Haiku] Compile error (attempt {attempt}): {last_error[:120]} — fixing..."})
                current_code = _autofix_signal_code(current_code, last_error, attempt, jid)
            else:
                raise RuntimeError(f"Signal code unfixable after {max_attempts} attempts. Last error: {last_error}") from exc
    raise RuntimeError("Unreachable")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY LIBRARY  — curated strategies with proven edge
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_LIBRARY = [
    # ── Momentum ────────────────────────────────────────────────────────────────
    {
        "id": "cs_momentum",
        "name": "Cross-Sectional Momentum (12-1)",
        "category": "momentum",
        "timeframe": "monthly",
        "complexity": "medium",
        "edge": "Winners keep winning — 12-month past returns predict next month's return",
        "description": "Jegadeesh & Titman (1993) seminal momentum anomaly. Rank assets by 12-month return skipping the last month, buy top quartile. The skip prevents short-term reversal contamination. Robust across US equities, international markets, sectors, and asset classes.",
        "indicators": ["12-1 month return rank", "monthly rebalance", "equal weight"],
        "expected_sharpe": "0.5–1.0",
        "expected_winrate": "52–58%",
        "best_conditions": "Bull markets, trending regimes, low correlation environments",
        "avoid_conditions": "Market crashes (momentum crashes), mean-reverting choppy markets",
        "sources": "Jegadeesh & Titman 1993 · AQR Momentum Factor (MOM)",
        "symbols": ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI"],
        "idea_prompt": "Cross-sectional momentum strategy based on Jegadeesh & Titman 1993. For a universe of ETFs (SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLI, XLU, XLP, XLY), rank by 12-month return skipping the most recent month (12-1 skip). Buy top 3 ranked ETFs monthly, sell bottom 3. Rebalance on the last trading day of each month. Include momentum score calculation (12-month log return minus 1-month return), ranking, position sizing, and monthly rebalancing logic."
    },
    {
        "id": "rsi_pullback",
        "name": "RSI-2 Pullback (Connors Research)",
        "category": "momentum",
        "timeframe": "daily",
        "complexity": "low",
        "edge": "Short-term mean reversion in strong uptrends — extreme oversold in bull markets bounce fast",
        "description": "Larry Connors' high-probability RSI strategy. Buy when 2-period RSI drops below 10 while price is above 200-day MA (confirming uptrend). Exit when RSI crosses above 70. Extremely high win rate in trending bull markets.",
        "indicators": ["RSI(2)", "SMA(200)", "price vs MA filter"],
        "expected_sharpe": "0.8–1.5",
        "expected_winrate": "65–75%",
        "best_conditions": "Bull markets, SPY above 200MA, low-volatility uptrends",
        "avoid_conditions": "Bear markets, downtrends, high VIX environments",
        "sources": "Larry Connors 'Short Term Trading Strategies That Work' 2008",
        "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"],
        "idea_prompt": "Larry Connors RSI-2 pullback strategy. Entry: 2-period RSI drops below 10 AND close is above 200-period SMA (uptrend filter). Exit: 2-period RSI rises above 70 OR close drops below 200-period SMA (stop). Also add a 5-period SMA exit as alternative. Position: full allocation per signal. This is a mean-reversion-within-uptrend strategy — only trade long side. Handle edge cases where RSI is NaN at start of series."
    },
    {
        "id": "dual_momentum",
        "name": "Dual Momentum (Antonacci)",
        "category": "momentum",
        "timeframe": "monthly",
        "complexity": "low",
        "edge": "Combines absolute + relative momentum — only invest in asset class with positive absolute momentum",
        "description": "Gary Antonacci's Dual Momentum (2014). Absolute momentum: only hold equities if SPY return > T-bill return over 12 months. Relative momentum: choose between US (SPY) or international (EFA) equities. When absolute momentum is negative, hold bonds (AGG). Very simple, very robust.",
        "indicators": ["12-month absolute momentum", "relative momentum", "T-bill rate"],
        "expected_sharpe": "0.7–1.2",
        "expected_winrate": "55–62%",
        "best_conditions": "Trending markets — avoids major bear markets via absolute momentum filter",
        "avoid_conditions": "Whipsaw markets that trigger false bear signals",
        "sources": "Gary Antonacci 'Dual Momentum Investing' 2014 · AQR research",
        "symbols": ["SPY", "EFA", "AGG", "BIL"],
        "idea_prompt": "Gary Antonacci Dual Momentum strategy. Monthly signal: (1) Absolute momentum test — compare SPY 12-month return vs BIL (T-bill proxy). (2) If SPY positive momentum: relative momentum test — compare SPY vs EFA 12-month return, buy whichever is stronger. (3) If SPY negative momentum: buy AGG (bonds). Rebalance monthly. This creates a regime-switching system that avoids major bear markets."
    },
    # ── Mean Reversion ──────────────────────────────────────────────────────────
    {
        "id": "bb_mean_reversion",
        "name": "Bollinger Band Mean Reversion",
        "category": "mean_reversion",
        "timeframe": "daily",
        "complexity": "low",
        "edge": "Price statistically reverts to mean after touching 2-sigma bands in ranging/low-volatility markets",
        "description": "Classic Bollinger Band (20,2) mean reversion. Buy when price touches or breaks lower band AND RSI(14) < 35 (oversold confirmation). Sell when price reaches middle band (20MA) or upper band. Add volume confirmation for quality entries.",
        "indicators": ["Bollinger Bands(20,2)", "RSI(14)", "Volume"],
        "expected_sharpe": "0.5–0.9",
        "expected_winrate": "55–65%",
        "best_conditions": "Low-volatility, range-bound markets; high mean-reversion environments",
        "avoid_conditions": "Strong trending markets (price walks down the lower band)",
        "sources": "John Bollinger 'Bollinger on Bollinger Bands' · Larry Connors research",
        "symbols": ["SPY", "QQQ", "GLD", "individual large-cap stocks"],
        "idea_prompt": "Bollinger Band mean reversion strategy. Entry: close touches or breaks below lower Bollinger Band (20-period, 2 std dev) AND RSI(14) below 35 AND volume above 1.2x 20-day average volume. Exit: close reaches 20-period SMA (middle band) OR RSI rises above 65. Add stop-loss at 2% below entry. Only trade when price is within 5% of 52-week high (quality filter to avoid value traps)."
    },
    {
        "id": "opening_range_breakout",
        "name": "Opening Range Breakout (ORB)",
        "category": "mean_reversion",
        "timeframe": "intraday",
        "complexity": "medium",
        "edge": "First 30-minute range sets institutional support/resistance; breakouts have directional conviction",
        "description": "Toby Crabel's Opening Range Breakout. Define the high/low of the first 30 minutes (opening range). Buy when price breaks above OR high with volume surge; short when below OR low. Target 1.5x the range, stop at OR boundary. Works best on high-volume liquid ETFs.",
        "indicators": ["Opening range H/L", "Volume surge", "ATR filter"],
        "expected_sharpe": "0.6–1.0",
        "expected_winrate": "52–58%",
        "best_conditions": "High-volume market open, catalyst days (earnings, Fed), trending days",
        "avoid_conditions": "Inside days, low-volume chop, options expiration",
        "sources": "Toby Crabel 'Day Trading with Short-Term Price Patterns' 1990",
        "symbols": ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"],
        "idea_prompt": "Opening Range Breakout strategy for 5-minute bars. Define opening range as the high and low of the first 6 bars (30 minutes). Buy signal: price breaks above opening range high with volume at least 1.5x average. Sell/exit signal: price drops below opening range low. Target: opening range high + 1.5 * (opening range high - opening range low). Stop: opening range midpoint. Only trade first 3 hours of session (first 36 bars on 5m). Reset daily."
    },
    {
        "id": "rsi_extremes",
        "name": "RSI Extremes Mean Reversion",
        "category": "mean_reversion",
        "timeframe": "daily",
        "complexity": "low",
        "edge": "Extreme RSI readings on large-cap stocks reliably predict short-term price recovery",
        "description": "Buy when RSI(14) drops below 20 (extreme oversold) on stocks above their 150-day MA. The MA filter ensures we buy quality stocks in dips, not falling knives. Exit when RSI recovers above 50. Combines momentum bias with short-term reversion.",
        "indicators": ["RSI(14)", "SMA(150)", "SMA(50)"],
        "expected_sharpe": "0.6–1.1",
        "expected_winrate": "60–68%",
        "best_conditions": "Bull market corrections, earnings gap-downs on strong companies",
        "avoid_conditions": "Sector-wide breaks, fundamental deterioration, bear markets",
        "sources": "Multiple academic papers on RSI mean reversion · IBD research",
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "idea_prompt": "RSI extreme mean reversion on large-cap quality stocks. Entry: RSI(14) drops below 20 AND close is above 150-period SMA (long-term uptrend) AND close is within 15% of 52-week high (avoiding value traps). Exit: RSI(14) rises above 55 OR close drops more than 7% below entry (stop). Position size: equal weight. This strategy identifies high-quality stocks having temporary pullbacks."
    },
    # ── Trend Following ─────────────────────────────────────────────────────────
    {
        "id": "turtle_donchian",
        "name": "Turtle Trading (Donchian Breakout)",
        "category": "trend",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "Systematic trend capture — small losses on false breakouts, large gains on real trends",
        "description": "Richard Dennis / Bill Eckhardt's Turtle Trading system. Buy on 20-day high breakout, sell on 10-day low. Exit rule: 2x ATR trailing stop. Position sizing by 1% risk per trade per 1 ATR. The key is strict discipline — many small losses and occasional large wins.",
        "indicators": ["Donchian Channel(20)", "Donchian Channel(10)", "ATR(20)"],
        "expected_sharpe": "0.4–0.8",
        "expected_winrate": "35–45% (wins are large vs losses)",
        "best_conditions": "Trending bull or bear markets; momentum regimes",
        "avoid_conditions": "Choppy sideways markets (many whipsaws reduce win rate to loss)",
        "sources": "Michael Covel 'The Complete Turtle Trader' 2007 · Original Turtle rules",
        "symbols": ["SPY", "QQQ", "GLD", "TLT", "USO", "commodity ETFs"],
        "idea_prompt": "Turtle Trading System adapted for ETFs. Entry: close breaks above 20-day Donchian Channel high (highest high of last 20 bars). Exit signals: (1) close drops below 10-day Donchian low (trailing channel stop), OR (2) close drops 2 ATR(20) below entry price (initial stop). Position sizing: risk 1% of capital per ATR move. Only enter if last trade was a loss (skip breakout if prior breakout was profitable — turtle filter). Handle ATR calculation properly."
    },
    {
        "id": "ma_trend_following",
        "name": "Triple Moving Average Trend",
        "category": "trend",
        "timeframe": "daily",
        "complexity": "low",
        "edge": "Three-MA alignment eliminates false signals from single crossover noise",
        "description": "Buy when 10MA > 30MA > 100MA (all aligned bullish) and price is above all three. Sell when 10MA crosses below 30MA. Using three MAs significantly reduces whipsaws compared to golden/death cross. Add ADX filter (ADX > 20) to confirm trend strength.",
        "indicators": ["SMA(10)", "SMA(30)", "SMA(100)", "ADX(14)"],
        "expected_sharpe": "0.5–0.9",
        "expected_winrate": "45–55%",
        "best_conditions": "Trending markets, low-chop environments, post-breakout trends",
        "avoid_conditions": "Sideways/ranging markets, high-volatility choppy periods",
        "sources": "Perry Kaufman 'Trading Systems and Methods' · Standard technical analysis",
        "symbols": ["SPY", "QQQ", "sector ETFs", "individual growth stocks"],
        "idea_prompt": "Triple Moving Average Trend Following strategy. Entry: SMA(10) > SMA(30) > SMA(100) all aligned bullishly AND close above all three MAs AND ADX(14) above 20 (confirming trend). Exit: SMA(10) crosses below SMA(30) OR close drops below SMA(100). Add position scaling: double position size when ADX > 30 (strong trend). Track the trend strength with ADX and reduce exposure in weak trends (ADX < 15)."
    },
    {
        "id": "keltner_trend",
        "name": "Keltner Channel Trend Rider",
        "category": "trend",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "Keltner channels adapt to volatility, giving cleaner trend entries than fixed MA crossovers",
        "description": "Buy when price closes above upper Keltner Channel (EMA + 2x ATR) — signals genuine trend breakout. Ride the trend while price stays above EMA. Exit when price closes below EMA for 2 consecutive days. The ATR-based width filters out false breakouts in low-volatility periods.",
        "indicators": ["EMA(20)", "ATR(10)", "Keltner Channel(20,2)"],
        "expected_sharpe": "0.5–0.85",
        "expected_winrate": "48–55%",
        "best_conditions": "Momentum-driven breakouts, earnings gaps, sector rotations",
        "avoid_conditions": "Mean-reverting environments, overbought conditions without catalyst",
        "sources": "Chester Keltner 1960 · Linda Bradford Raschke adaptations",
        "symbols": ["SPY", "QQQ", "NVDA", "TSLA", "growth stocks"],
        "idea_prompt": "Keltner Channel trend breakout strategy. Calculate EMA(20) and ATR(10). Upper channel = EMA + 2*ATR, Lower channel = EMA - 2*ATR. Entry: close breaks above upper Keltner Channel for first time after being below it. Exit: close drops below EMA(20) for 2 consecutive days. Momentum confirmation: Volume on breakout day must be above 20-day average. Stop: 2 ATR below entry price."
    },
    # ── Volatility / Regime ──────────────────────────────────────────────────────
    {
        "id": "vix_regime",
        "name": "VIX Regime Switching (Fear Filter)",
        "category": "volatility",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "VIX above 30 predicts poor equity returns — switching to bonds/cash avoids major drawdowns",
        "description": "Regime switching strategy using VIX as fear gauge. Hold SPY when VIX < 20 (calm). Move to TLT (bonds) when VIX 20-30 (caution). Move to cash or inverse ETF when VIX > 30 (fear). Rebalance daily. The key insight: high-VIX environments have negative Sharpe for buy-and-hold equities.",
        "indicators": ["VIX level", "VIX 20-day MA", "SPY trend"],
        "expected_sharpe": "0.7–1.3",
        "expected_winrate": "55–65%",
        "best_conditions": "Works in all regimes — the whole point is regime detection",
        "avoid_conditions": "VIX manipulation events, very short-lived spikes (COVID spike was only 2 weeks)",
        "sources": "Academic: VIX as fear gauge (Whaley 2000) · Systematic research",
        "symbols": ["SPY", "TLT", "BIL", "^VIX"],
        "idea_prompt": "VIX regime-switching strategy. Use VIX closing level as regime indicator. When VIX < 20: fully invested in SPY (risk-on). When VIX 20-25: reduce SPY to 50%, add TLT 50% (cautious). When VIX 25-30: 25% SPY, 75% TLT (defensive). When VIX > 30: exit SPY, hold TLT or cash. Add smoothing: use 3-day VIX average to avoid daily whipsaws. Signal is set at close, executed at next open. Note: implement using ^VIX data alongside SPY."
    },
    {
        "id": "vol_breakout",
        "name": "Volatility Contraction Breakout",
        "category": "volatility",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "Low volatility periods compress price, creating energy for explosive directional moves",
        "description": "Mark Minervini's Volatility Contraction Pattern (VCP). Identify price consolidations where trading range shrinks for 3+ consecutive contractions. Enter on the final, tightest pivot breakout with volume surge. This is used by many top CANSLIM traders.",
        "indicators": ["ATR(14)", "Bollinger Band width", "Volume", "Pivot high"],
        "expected_sharpe": "0.7–1.4",
        "expected_winrate": "45–55% (but avg win >> avg loss)",
        "best_conditions": "Bull market, leading sectors, fundamentally strong stocks near 52-week highs",
        "avoid_conditions": "Bear markets, low-quality stocks, low-volume breakouts",
        "sources": "Mark Minervini 'Trade Like a Stock Market Wizard' 2013 · CANSLIM",
        "symbols": ["High RS growth stocks near 52-week highs", "NVDA", "TSLA", "SMCI"],
        "idea_prompt": "Volatility Contraction Pattern (VCP) breakout strategy. Identify consolidations: 3 consecutive weeks where weekly high-low range contracts. Entry trigger: price breaks above the pivot high on volume at least 40% above 50-day average. Measure contraction by comparing current ATR(14) to 30-day ATR average — enter when current ATR < 0.7 * 30-day ATR (tight consolidation). Stop: 7-8% below entry (Minervini's stop). Target: 20-25% or trailing stop. Confirm: stock within 25% of 52-week high."
    },
    # ── Factor / Value ───────────────────────────────────────────────────────────
    {
        "id": "low_vol_anomaly",
        "name": "Low Volatility Anomaly",
        "category": "value",
        "timeframe": "monthly",
        "complexity": "medium",
        "edge": "Low-beta stocks consistently outperform high-beta stocks on risk-adjusted basis (inverted CAPM)",
        "description": "Baker, Bradley & Wurgler (2011) showed low-volatility stocks outperform despite lower risk — the opposite of what CAPM predicts. The anomaly persists due to benchmark-driven investing. Buy lowest-volatility quartile of S&P 500, rebalance monthly.",
        "indicators": ["20-day realized volatility", "Beta to SPY", "monthly ranking"],
        "expected_sharpe": "0.7–1.1",
        "expected_winrate": "55–60%",
        "best_conditions": "Risk-off environments; defensive equity rotations",
        "avoid_conditions": "Momentum rallies, risk-on periods (low-vol underperforms when markets sprint)",
        "sources": "Baker, Bradley & Wurgler 2011 · MSCI Min Volatility Factor",
        "symbols": ["SPLV ETF", "USMV ETF", "XLP", "XLU", "XLV"],
        "idea_prompt": "Low Volatility Anomaly factor strategy. For a universe of ETFs (SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLU, XLP, XLY, GLD, TLT), calculate 20-day realized volatility for each. Rank by volatility (lowest to highest). Buy the 3 lowest-volatility ETFs each month. Equal weight. Monthly rebalance. Also add a quality filter: exclude ETFs that have declined more than 20% in past 3 months (avoid value traps). Compare performance against SPY benchmark."
    },
    {
        "id": "sector_rotation",
        "name": "Sector Momentum Rotation",
        "category": "value",
        "timeframe": "monthly",
        "complexity": "medium",
        "edge": "Sector performance has strong momentum — outperforming sectors continue for 3-12 months",
        "description": "Rotate monthly into the top 3 performing SPDR sector ETFs based on 3-month and 6-month momentum. Equal-weight top 3. Move allocation out of lagging sectors. Faber (2007) showed momentum rotation on sectors generates Sharpe > 1.0 historically.",
        "indicators": ["3-month return", "6-month return", "momentum composite"],
        "expected_sharpe": "0.6–1.0",
        "expected_winrate": "53–60%",
        "best_conditions": "Trending markets with clear sector leadership",
        "avoid_conditions": "Sector rotations that reverse quickly (pandemic crashes/recoveries)",
        "sources": "Meb Faber 'A Quantitative Approach to Tactical Asset Allocation' 2007",
        "symbols": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"],
        "idea_prompt": "Sector Momentum Rotation strategy using all 11 SPDR sector ETFs. Monthly signal: rank each sector ETF by composite momentum score = 0.5 * 3-month return + 0.5 * 6-month return. Buy top 3 ranked sectors, equal weight (33.3% each). Exit any sector not in top 3. Only hold a sector if it has positive absolute momentum (positive composite score). If fewer than 3 sectors qualify, hold cash for the remaining allocation. Rebalance on last trading day of month."
    },
    # ── Macro / Multi-asset ─────────────────────────────────────────────────────
    {
        "id": "risk_on_off",
        "name": "Risk-On / Risk-Off (SPY vs TLT)",
        "category": "macro",
        "timeframe": "daily",
        "complexity": "low",
        "edge": "SPY and TLT are negatively correlated in risk events — systematic switching captures both trends",
        "description": "Simple regime model: buy SPY when SPY > 200MA (risk-on), switch to TLT when SPY < 200MA (risk-off). Enhancement: use the 10-month SMA (Faber 2007) for monthly version. Dramatically reduces max drawdown vs buy-and-hold with only marginal return reduction.",
        "indicators": ["SMA(200) or SMA(10-month)", "SPY trend", "TLT as safe haven"],
        "expected_sharpe": "0.6–1.0",
        "expected_winrate": "55–65%",
        "best_conditions": "Trend-following environments — captures both bull and bear regimes",
        "avoid_conditions": "Quick V-shape reversals (March 2020) where system exits and misses recovery",
        "sources": "Meb Faber 2007 · Standard tactical asset allocation research",
        "symbols": ["SPY", "TLT", "BIL"],
        "idea_prompt": "Risk-On/Risk-Off tactical asset allocation strategy. Daily signal: when SPY close is above its 200-day SMA, hold SPY (risk-on mode). When SPY close is below its 200-day SMA, switch to TLT (risk-off mode). Add enhancement: use RSI(14) of SPY as secondary signal — if SPY > 200MA but RSI < 30 (oversold), hold position (avoid selling at short-term panic lows). Rebalance when signal changes. Track regime and calculate time in each regime."
    },
    {
        "id": "gold_spy_hedge",
        "name": "SPY/GLD Dynamic Allocation",
        "category": "macro",
        "timeframe": "weekly",
        "complexity": "low",
        "edge": "Gold hedges inflation and tail risk — dynamic allocation outperforms static 60/40 on risk-adjusted basis",
        "description": "Allocate between SPY and GLD based on relative momentum and macro signals. When real yields are falling (bond prices rising), increase GLD allocation. When SPY momentum is strong, increase SPY allocation. Rebalance weekly. Reduces correlation to equity drawdowns.",
        "indicators": ["SPY momentum", "GLD momentum", "TLT trend (real yields proxy)"],
        "expected_sharpe": "0.6–1.0",
        "expected_winrate": "52–58%",
        "best_conditions": "Inflationary environments, geopolitical uncertainty, late business cycle",
        "avoid_conditions": "Strong USD periods, rising real yields compress gold",
        "sources": "Ray Dalio All-Weather · Permanent Portfolio (Harry Browne)",
        "symbols": ["SPY", "GLD", "TLT", "BIL"],
        "idea_prompt": "SPY/GLD Dynamic Allocation strategy. Weekly signal: calculate 3-month momentum for SPY and GLD. Allocate proportionally to relative momentum: SPY weight = SPY_momentum / (SPY_momentum + GLD_momentum), GLD weight = remainder. Add defensive overlay: if SPY is below 200-day MA, cap SPY at 30% and fill remainder with TLT. If both SPY and GLD are in downtrends (both below 50MA), hold 50% TLT, 50% BIL (cash). Rebalance weekly."
    },
    # ── Statistical Arbitrage ────────────────────────────────────────────────────
    {
        "id": "pairs_spy_qqq",
        "name": "SPY/QQQ Pairs Trade",
        "category": "arb",
        "timeframe": "daily",
        "complexity": "high",
        "edge": "SPY and QQQ are cointegrated — their spread reverts to mean, exploitable with z-score signals",
        "description": "Classic pairs trade on SPY (broad market) vs QQQ (tech-heavy). Calculate the spread ratio and its z-score over a rolling 60-day window. Buy SPY/sell QQQ when z-score < -2 (QQQ overperformed), reverse when z-score > +2. Market-neutral: no directional market exposure.",
        "indicators": ["Price ratio", "Z-score(60)", "cointegration test"],
        "expected_sharpe": "0.5–1.2",
        "expected_winrate": "58–68%",
        "best_conditions": "Range-bound markets; periods when tech diverges from broad market",
        "avoid_conditions": "Strong sustained tech momentum (ratio trends); structural breaks",
        "sources": "Gatev, Goetzmann & Rouwenhorst 1999 · Statistical arbitrage literature",
        "symbols": ["SPY", "QQQ"],
        "idea_prompt": "SPY/QQQ pairs trading strategy. Calculate the price ratio SPY/QQQ. Compute rolling 60-day z-score of this ratio: z = (ratio - rolling_mean) / rolling_std. Entry long SPY / short QQQ: z-score < -2.0 (QQQ overextended vs SPY). Entry long QQQ / short SPY: z-score > +2.0 (SPY overextended vs QQQ). Exit: z-score returns to 0.5 or within 0.5 of mean. Stop: z-score extends to ±3.5 (pair diverging further). In this simplified backtest, implement as a single long/short signal on SPY only."
    },
    # ── Seasonal ─────────────────────────────────────────────────────────────────
    {
        "id": "sell_in_may",
        "name": "Sell in May (Halloween Effect)",
        "category": "seasonal",
        "timeframe": "monthly",
        "complexity": "low",
        "edge": "Nov-April returns significantly exceed May-Oct returns across global markets",
        "description": "Bouman & Jacobsen (2002) documented globally. Hold equities Nov 1 to Apr 30, switch to bonds or cash May 1 to Oct 31. The effect persists across 36 countries over 300+ years of data. Simple, robust, requires minimal monitoring.",
        "indicators": ["Calendar month", "Seasonal pattern"],
        "expected_sharpe": "0.5–0.9 vs buy-and-hold",
        "expected_winrate": "60–70% of years outperform buy-and-hold",
        "best_conditions": "All market regimes — the seasonal pattern is remarkably persistent",
        "avoid_conditions": "Strong bull years can have good May-Oct returns — opportunity cost",
        "sources": "Bouman & Jacobsen 2002 · Ben Jacobsen ongoing research (300+ year dataset)",
        "symbols": ["SPY", "QQQ", "TLT", "BIL"],
        "idea_prompt": "Sell in May seasonal strategy. Long SPY from November 1 through April 30 (6 months). Hold TLT from May 1 through October 31 (6 months). Enhanced version: during May-October, switch from TLT to SPY if SPY closes above 200MA for 5 consecutive days (trending filter). Switch back to TLT if that condition fails. This captures the seasonal effect while not missing major bull runs. Calculate signals based on month of year."
    },
    {
        "id": "january_effect",
        "name": "Small Cap January Effect",
        "category": "seasonal",
        "timeframe": "monthly",
        "complexity": "low",
        "edge": "Tax-loss selling in December creates bargains in small caps; January buying pressure follows",
        "description": "Buy IWM (small cap ETF) in mid-to-late December when tax-loss selling peaks. Sell by end of January. Enhancement: the effect is strongest in stocks that dropped 20-40% during the year (max tax-loss candidates). The last 2 weeks of December and first 2 weeks of January are prime.",
        "indicators": ["Calendar timing", "YTD return ranking", "small cap vs large cap spread"],
        "expected_sharpe": "0.6–1.2 (concentrated in 1 month)",
        "expected_winrate": "65–75% of years",
        "best_conditions": "After market declines, when small cap value stocks have underperformed",
        "avoid_conditions": "Already overbought small caps; reduced by increased awareness of effect",
        "sources": "Keim 1983 · Rozeff & Kinney 1976 · Standard seasonal literature",
        "symbols": ["IWM", "VBR", "SCHA"],
        "idea_prompt": "Small Cap January Effect seasonal strategy. Go long IWM starting December 15 (tax-loss selling pressure peaks). Hold through January 31. Exit February 1. During the holding period, use a momentum filter: only hold if IWM is above its 50-day SMA on December 15 entry date. If IWM is below 50MA (bearish context), reduce position to 50%. Compare returns against SPY benchmark. Track years where the strategy worked vs failed."
    },
    # ── Advanced / Quant ────────────────────────────────────────────────────────
    {
        "id": "carry_trade",
        "name": "Equity Carry (Dividend + Buyback Yield)",
        "category": "value",
        "timeframe": "monthly",
        "complexity": "high",
        "edge": "High shareholder yield (dividends + buybacks) predicts superior future returns",
        "description": "Meb Faber's Shareholder Yield strategy. Rank stocks by total shareholder yield = dividend yield + net buyback yield. Buy top decile. Avoid yield traps by adding momentum filter. Works because companies returning cash have discipline and fair valuations.",
        "indicators": ["Dividend yield", "Buyback yield", "Shareholder yield rank", "12-month momentum filter"],
        "expected_sharpe": "0.6–1.0",
        "expected_winrate": "54–60%",
        "best_conditions": "Value regimes, late business cycle, when growth trades at extreme premium",
        "avoid_conditions": "QE-driven growth rallies; tech bubble environments",
        "sources": "Meb Faber 'Shareholder Yield' · Siegel dividend research",
        "symbols": ["DVY", "SCHD", "VYM", "high yield ETFs"],
        "idea_prompt": "Shareholder Yield momentum-enhanced strategy. Universe: high-dividend ETFs and sector ETFs (DVY, SCHD, VYM, XLP, XLU, XLV, XLE, XLF). Rank by proxy yield (use 12-month price decline as yield proxy since dividend data is limited in yfinance). Apply momentum filter: only buy ETFs with positive 6-month momentum. Buy top 3 by composite score (40% yield proxy + 60% momentum). Monthly rebalance. Add quality filter: exclude ETFs with negative 12-month return (avoid value traps)."
    },
    {
        "id": "trend_quality",
        "name": "Trend Quality Score (ADX + Volume)",
        "category": "trend",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "ADX + volume expansion confirms real institutional buying vs noise — fewer false signals",
        "description": "Enter trends only when quality score is high: ADX > 25 (strong trend), ADX slope rising (accelerating), volume 20% above average (institutional participation), and price above both 20MA and 50MA. This filters out low-quality setups and improves win rate significantly.",
        "indicators": ["ADX(14)", "DI+/DI-", "Volume MA", "SMA(20)", "SMA(50)"],
        "expected_sharpe": "0.6–1.1",
        "expected_winrate": "52–60%",
        "best_conditions": "Strong trending markets with clear institutional conviction",
        "avoid_conditions": "Low-ADX chop; low-volume drift upward",
        "sources": "Welles Wilder ADX (1978) · IBD volume analysis",
        "symbols": ["SPY", "QQQ", "sector leaders", "high-RS stocks"],
        "idea_prompt": "Trend Quality Score strategy. Calculate: ADX(14), +DI(14), -DI(14), SMA(20), SMA(50), 20-day volume average. Entry: ADX > 25 AND +DI > -DI (bullish trend) AND ADX increasing (ADX today > ADX 3 days ago) AND close > SMA(20) > SMA(50) AND volume > 1.2 * 20-day volume average. Exit: ADX drops below 20 (trend weakening) OR close drops below SMA(50). Stop: 2% below SMA(20). The quality score is ADX * (volume_ratio) — only trade top quality setups."
    },
    {
        "id": "earnings_momentum",
        "name": "Post-Earnings Announcement Drift (PEAD)",
        "category": "momentum",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "Stocks that beat earnings estimates continue drifting higher for 60 days after announcement",
        "description": "Bernard & Thomas (1989) seminal anomaly. Positive earnings surprises cause underreaction — price drifts up for 30-60 days. Proxy: buy stocks that gap up 3%+ on high volume on earnings day. The gap signals a big beat. Hold 20-30 trading days.",
        "indicators": ["Gap-up on earnings", "Volume surge", "Price vs 50MA"],
        "expected_sharpe": "0.7–1.3",
        "expected_winrate": "55–65%",
        "best_conditions": "Bull markets; tech and growth earnings beats",
        "avoid_conditions": "Bear markets (even beats fade); sector headwinds",
        "sources": "Bernard & Thomas 1989 · Multiple PEAD replications",
        "symbols": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
        "idea_prompt": "Post-Earnings Announcement Drift (PEAD) strategy. Identify earnings gaps: a day where open is 3%+ above previous close on volume at least 2x average. Buy signal: gap-up day with volume surge (PEAD entry). Hold for 20 trading days (approximately 1 month). Exit: 20 days elapsed OR close drops below 50-day SMA OR drops 8% below entry (stop). Only take gaps that occur when stock is already in uptrend (above 200MA). Implement as: if today's open/close shows gap-up pattern and volume > 2x average, generate buy signal."
    },
    {
        "id": "crypto_momentum",
        "name": "Crypto Momentum + RSI",
        "category": "momentum",
        "timeframe": "daily",
        "complexity": "medium",
        "edge": "Crypto has stronger momentum factor than equities — 3-month winners continue for 1-3 months",
        "description": "Bitcoin/Ethereum tend to have strong momentum periods followed by sharp reversals. Buy BTC/ETH when 90-day momentum is positive AND RSI(14) 40-65 (not overbought). Exit when RSI > 75 (overbought) or 90-day momentum turns negative. Crypto momentum is well-documented since 2017.",
        "indicators": ["90-day return", "RSI(14)", "SMA(50)", "SMA(200)"],
        "expected_sharpe": "0.5–1.5 (high variance)",
        "expected_winrate": "50–58%",
        "best_conditions": "Bitcoin bull cycles, risk-on environments, post-halving periods",
        "avoid_conditions": "Bear markets, regulatory crackdowns, high-leverage unwind events",
        "sources": "Liu & Tsyvinski 2021 'Risks and Returns of Cryptocurrency' · Factor research",
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "idea_prompt": "Crypto momentum with RSI filter for BTC/ETH/SOL. Entry: 90-day return is positive (positive momentum) AND RSI(14) is between 35-65 (not overbought, not oversold) AND close above SMA(50). Exit: RSI(14) rises above 75 (overbought, take profit) OR 90-day return turns negative (momentum breakdown) OR close drops below SMA(50). Position size: 50% on first signal, add 25% if momentum continues after 2 weeks. Crypto-specific: use daily data, handle weekend trading (no gaps)."
    },
    {
        "id": "volatility_risk_premium",
        "name": "Volatility Risk Premium (VRP) Harvest",
        "category": "volatility",
        "timeframe": "weekly",
        "complexity": "high",
        "edge": "Implied volatility (VIX) consistently exceeds realized volatility — selling fear premium profits on average",
        "description": "The implied volatility of options (VIX) exceeds realized volatility ~80% of the time. This VRP is harvestable by being short volatility when VIX > 20 (elevated fear premium). Proxy using SVXY (short VIX) or VXX short signals. Must have strict drawdown limits.",
        "indicators": ["VIX level", "VIX vs realized vol spread", "VIX term structure"],
        "expected_sharpe": "0.8–1.5 (with strict risk management)",
        "expected_winrate": "65–75% (but losers are large)",
        "best_conditions": "VIX 15-25, positive term structure (contango), calm markets",
        "avoid_conditions": "VIX spikes > 30, inverted term structure, tail risk events",
        "sources": "Carr & Wu 2009 VRP research · Volatility risk premium literature",
        "symbols": ["SPY", "^VIX proxy", "VXX as inverse signal"],
        "idea_prompt": "Volatility Risk Premium harvest strategy (equity proxy). When VIX is elevated (20-30), the market overestimates future volatility — mean reversion in equities is more likely. Strategy: when VIX > 20, buy SPY (the fear creates buying opportunities). When VIX drops below 15, reduce exposure (low fear = less risk premium available). Position size inversely proportional to VIX: size = base * (1 + (VIX-15)/20). Add VIX momentum filter: only increase exposure when VIX is declining (VIX today < VIX 5 days ago)."
    },
]


_SUGGEST_SYSTEM = """\
You are a senior quantitative researcher helping a trader find the best algorithmic strategies for their situation.
Given a curated strategy library and the trader's specific requirements, rank the TOP 5 most suitable strategies.

Consider:
- Market conditions described (bull/bear, volatility, sector)
- Trader's experience and risk tolerance
- Timeframe preference
- Specific requirements or constraints

Return JSON:
{
  "suggestions": [
    {
      "strategy_id": "id from library",
      "rank": 1,
      "fit_score": 8.5,
      "why_fits": "2-sentence explanation of why this strategy fits their requirements",
      "key_edge": "the specific edge this provides in their context",
      "caveats": "one key risk or caveat to watch for"
    }
  ],
  "market_note": "brief overall market context advice (1-2 sentences)",
  "best_timeframe": "which timeframes look most attractive right now"
}
Return only valid JSON."""


def suggest_strategies(query: str, market_context: str) -> dict:
    """Use DeepSeek to rank library strategies for a trader's specific needs."""
    library_summary = "\n".join(
        f"- id={s['id']} | {s['name']} | category={s['category']} | timeframe={s['timeframe']} "
        f"| edge: {s['edge']} | Sharpe: {s['expected_sharpe']} | conditions: {s['best_conditions']}"
        for s in STRATEGY_LIBRARY
    )
    prompt = (
        f"TRADER REQUEST:\n{query}\n\n"
        f"MARKET CONTEXT:\n{market_context or 'Not specified — assume current typical market conditions'}\n\n"
        f"AVAILABLE STRATEGIES:\n{library_summary}\n\n"
        "Rank the top 5 most suitable strategies for this trader."
    )
    raw = _grok(_SUGGEST_SYSTEM, prompt, max_tokens=1500)
    try:
        return _extract_json(raw)
    except Exception:
        return {"suggestions": [], "market_note": raw[:300]}


def get_strategy_library() -> list[dict]:
    """Return the full strategy library (without verbose idea_prompt for listing)."""
    return [
        {k: v for k, v in s.items() if k != "idea_prompt"}
        for s in STRATEGY_LIBRARY
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH — Reddit + DuckDuckGo strategy discovery
# ═══════════════════════════════════════════════════════════════════════════════

_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 AlphaTerminal/1.0 (algo-trading research; educational use)"
}

_SUBREDDITS = ["algotrading", "quantfinance", "stocks", "Daytrading", "investing"]


def _search_reddit(query: str) -> list[dict]:
    results = []
    for sub in _SUBREDDITS[:3]:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            resp = _requests.get(
                url,
                params={"q": query, "sort": "top", "t": "year", "limit": 6, "restrict_sr": "true"},
                headers=_SEARCH_HEADERS, timeout=8,
            )
            if not resp.ok:
                continue
            for post in resp.json().get("data", {}).get("children", []):
                p = post.get("data", {})
                title = p.get("title", "")
                body  = (p.get("selftext") or "")[:600]
                if not title:
                    continue
                results.append({
                    "source": f"Reddit r/{sub}",
                    "title":  title,
                    "text":   body,
                    "url":    "https://reddit.com" + p.get("permalink", ""),
                    "score":  p.get("score", 0),
                })
        except Exception:
            pass
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


def _search_duckduckgo(query: str) -> list[dict]:
    """Full-text DuckDuckGo search via duckduckgo_search package (optional dep)."""
    try:
        from duckduckgo_search import DDGS  # type: ignore
        results = []
        with DDGS() as ddg:
            for r in ddg.text(query + " algo trading strategy python", max_results=8):
                results.append({
                    "source": "Web",
                    "title":  r.get("title", ""),
                    "text":   r.get("body", "")[:500],
                    "url":    r.get("href", ""),
                })
        return results
    except Exception:
        return []


_WEB_EXTRACT_SYSTEM = """\
You are a practical algorithmic trading advisor extracting IMPLEMENTABLE strategy ideas
from web search results for RETAIL algo-traders using Python + yfinance.

From the search results, identify 3–5 distinct trading strategies that:
- Can be coded in Python with freely available data (yfinance, pandas)
- Are relevant to retail traders ($5k–$100k accounts)
- Have a clear, logical trading edge

For each strategy return enough detail to implement it. Ignore vague or purely theoretical ideas.

Return JSON:
{
  "strategies": [
    {
      "name": "short strategy name",
      "source": "Reddit r/algotrading or Web",
      "source_url": "url if available",
      "category": "momentum|mean_reversion|trend|volatility|value|seasonal|macro|arb",
      "timeframe": "daily|hourly|weekly|monthly",
      "complexity": "low|medium|high",
      "edge": "one sentence — why does this make money?",
      "description": "2–3 sentences describing the strategy clearly",
      "indicators": ["list", "of", "indicators"],
      "entry_rules": "specific entry conditions",
      "exit_rules": "specific exit conditions",
      "symbols": ["SPY", "QQQ", "etc"],
      "min_capital": "estimated minimum USD",
      "idea_prompt": "detailed prompt for code generation — include all rules, indicators, entry/exit"
    }
  ],
  "summary": "1–2 sentence overview of what you found"
}
Return only valid JSON."""


def web_search_strategies(query: str) -> dict:
    """Search Reddit + web for trading strategies and extract actionable ideas."""
    reddit  = _search_reddit(query)
    web     = _search_duckduckgo(query)
    all_res = reddit + web

    if not all_res:
        return {
            "strategies": [],
            "summary": "No results found. Reddit may be rate-limiting — try again in a moment.",
        }

    formatted = "\n\n---\n\n".join(
        f"[{r['source']}] {r['title']}\nURL: {r['url']}\n{r['text']}"
        for r in all_res[:12]
    )
    prompt = (
        f"Search query: '{query}'\n\n"
        f"Search results:\n{formatted}\n\n"
        "Extract 3–5 specific, implementable trading strategies from these results."
    )
    raw = _grok(_WEB_EXTRACT_SYSTEM, prompt, max_tokens=2500)
    try:
        return _extract_json(raw)
    except Exception:
        return {"strategies": [], "summary": raw[:400]}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STRATEGY RESEARCH & GENERATION  (Workflow 2)
# ═══════════════════════════════════════════════════════════════════════════════

_RESEARCH_SYSTEM = """\
You are a senior quantitative researcher at a top hedge fund with deep knowledge of
technical analysis, quantitative finance, academic research, and professional trading.

Given a trading idea or theme, produce a comprehensive strategy concept.
Return JSON with:
{
  "strategy_name": "concise name",
  "rationale": "why this edge exists in markets (2-3 paragraphs)",
  "market_conditions": "when this works best",
  "risk_factors": "key risks and failure modes",
  "indicators": ["list of indicators/signals to use"],
  "entry_rules": "specific measurable entry conditions",
  "exit_rules": "specific measurable exit conditions",
  "position_sizing": "how to size positions",
  "risk_management": "stop-loss and risk controls",
  "timeframe": "daily/weekly/etc",
  "universe": ["suggested ticker symbols or ETFs to trade"],
  "expected_edge": "quantified expected edge (e.g. 55% win rate, 1.5 profit factor)",
  "weaknesses": ["known weaknesses of this approach"]
}
Return only valid JSON."""

_CODEGEN_SYSTEM = """\
You are a senior quant developer. Convert a trading strategy concept into production-ready Python.
The function must be named generate_signals(df) and accept a pandas DataFrame with columns:
Open, High, Low, Close, Volume (all properly capitalized).
It must return a pd.Series with the same index containing:
  1 = buy/long signal, -1 = sell/exit signal, 0 = hold/flat
Rules:
- No look-ahead bias (never use future data)
- Use only pandas and numpy (no external libraries unless critical)
- Add brief inline comments explaining each signal step
- Handle edge cases (NaN, insufficient data) gracefully
- The function must work on both daily and intraday data

Return ONLY the Python function code, no markdown fences, no explanation."""


def research_idea(idea: str, jid: str) -> dict:
    _upd(jid, step="Researching trading idea...", **{"log+": f"[DeepSeek] Researching: {idea[:80]}"})
    raw = _grok(_RESEARCH_SYSTEM, f"Research this trading idea and create a strategy concept:\n\n{idea}",
                max_tokens=2000, jid=jid)
    return _extract_json(raw)


def generate_strategy_code(concept: dict, jid: str) -> str:
    _upd(jid, step="Generating strategy code...", **{"log+": "[Haiku] Generating Python signal code"})
    prompt = (
        f"Convert this strategy concept into Python code:\n\n"
        f"Strategy: {concept.get('strategy_name')}\n"
        f"Entry: {concept.get('entry_rules')}\n"
        f"Exit: {concept.get('exit_rules')}\n"
        f"Indicators: {', '.join(concept.get('indicators', []))}\n"
        f"Risk: {concept.get('risk_management')}\n\n"
        f"Full concept:\n{json.dumps(concept, indent=2)}"
    )
    return _haiku(_CODEGEN_SYSTEM, prompt, max_tokens=1500, jid=jid)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STRATEGY PARSING & ISSUE DETECTION  (Workflow 1)
# ═══════════════════════════════════════════════════════════════════════════════

_PARSE_SYSTEM = """\
You are a senior quant developer. Fully analyze the provided trading strategy code.
Return JSON with:
{
  "strategy_name": "short name",
  "description": "what the strategy does (2-3 sentences)",
  "indicators": ["indicators used"],
  "entry_logic": "plain English",
  "exit_logic": "plain English",
  "risk_management": "stop-loss, sizing etc",
  "timeframe": "daily/weekly/intraday",
  "interval": "yfinance string: 1m/5m/15m/30m/1h/1d/1wk/1mo",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "symbols": ["tickers found in code or []"],
  "signal_generator": "complete generate_signals(df) function — no look-ahead bias, handles NaN, returns pd.Series of 1/-1/0"
}
Return only valid JSON."""

_ISSUE_SYSTEM = """\
You are a practical algorithmic trading advisor reviewing strategies for RETAIL algo-traders.
Your audience: individual traders with $5k–$100k, using Interactive Brokers / Alpaca / Webull,
coding in Python with yfinance, running strategies on a home computer or small VPS.

Flag REAL problems only — not theoretical hedge-fund concerns. Focus on:
- Look-ahead bias (using future data — this is a real bug, always flag it)
- Obvious code bugs that would prevent execution
- Completely unrealistic assumptions (e.g. zero slippage on illiquid stocks)
- Strategies that require expensive data or institutional-only infrastructure

Do NOT flag as issues:
- Small sample size concerns unless under 6 months of data
- Low trade count (retail traders may have 10-50 trades/year — that's fine)
- Sharpe below 1.0 (a 0.4 Sharpe beats most mutual funds — it's tradeable)
- Missing factor exposures or portfolio construction theory

Return JSON:
{
  "issues": [
    {
      "type": "Look-ahead Bias|Code Bug|Unrealistic Cost|Data Problem|Logic Error|Other",
      "severity": "Critical|High|Medium|Low",
      "description": "plain English — what the issue is",
      "location": "which line/variable/section",
      "fix": "specific fix a retail trader can actually implement"
    }
  ],
  "overall_quality": "Poor|Fair|Good|Excellent",
  "retail_viable": true/false,
  "min_capital_usd": 5000,
  "automation_difficulty": "Easy|Medium|Hard",
  "broker_compatibility": "Any|IBKR/Alpaca|Institutional only",
  "survivorship_bias_risk": "High|Medium|Low",
  "implementation_ready": true/false,
  "summary": "2-sentence practical assessment for a retail trader"
}
Return only valid JSON."""


def parse_strategy(code: str, jid: str) -> dict:
    _upd(jid, step="Parsing strategy...", **{"log+": "[DeepSeek] Parsing strategy code"})
    raw = _grok(_PARSE_SYSTEM,
                f"Parse this strategy:\n\n```python\n{code}\n```",
                max_tokens=2500, jid=jid)
    return _extract_json(raw)


def detect_issues(code: str, jid: str) -> dict:
    _upd(jid, step="Detecting issues...", **{"log+": "[DeepSeek] Running issue detection"})
    raw = _grok(_ISSUE_SYSTEM,
                f"Analyze this strategy for issues:\n\n```python\n{code}\n```",
                max_tokens=2000, jid=jid)
    try:
        return _extract_json(raw)
    except Exception:
        return {"issues": [], "overall_quality": "Unknown", "summary": raw[:500]}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-AGENT DEBATE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

_RETAIL_CONTEXT = """\

IMPORTANT RETAIL CONTEXT: This strategy is for an individual retail algo-trader, NOT a hedge fund.
- Account size: typically $5k–$100k
- Broker: Interactive Brokers, Alpaca, Webull, or similar retail
- Data source: yfinance (free), no Bloomberg/Reuters
- Execution: automated Python script, 1-5 second fill latency
- Acceptable Sharpe: anything above 0.3 beats most passive investors on risk-adjusted basis
- Acceptable win rate: 45%+ is fine if the profit factor > 1.2
- A strategy with 15 trades/year and Sharpe 0.5 is WORTH TRADING for retail
Be practical and encouraging where warranted. Max 300 words."""

_AGENTS = [
    {
        "name": "Quant Researcher",
        "emoji": "🔬",
        "system": (
            "You are a quantitative researcher who advises both institutional and retail traders. "
            "Identify whether there is a real statistical edge — but calibrate expectations for retail: "
            "a Sharpe of 0.3–0.8 with consistent positive expectancy is genuinely valuable. "
            "Flag data mining and look-ahead bias specifically. Be constructive, not dismissive."
            + _RETAIL_CONTEXT
        ),
        "focus": "Evaluate the statistical edge, overfitting risk, and whether the edge is real and durable.",
    },
    {
        "name": "Retail Trader",
        "emoji": "💰",
        "system": (
            "You are an experienced retail algo-trader who has been running automated strategies "
            "for 8 years on Interactive Brokers and Alpaca with accounts of $10k–$200k. "
            "You care deeply about: can I actually run this at home? What does slippage really cost? "
            "Is the signal frequency manageable? Can I implement it with yfinance + Python? "
            "Be honest but realistic — retail strategies don't need hedge-fund-level Sharpe ratios."
            + _RETAIL_CONTEXT
        ),
        "focus": "Assess real-world retail execution: slippage, broker support, data needs, automation complexity, minimum capital.",
    },
    {
        "name": "Statistician",
        "emoji": "📊",
        "system": (
            "You are a practical statistician focused on trading system robustness. "
            "Check for the most important issues: look-ahead bias, parameter overfit, regime dependency. "
            "For retail traders: 1–3 years of backtest data may be all that's available — "
            "flag if results rely on a single lucky period, but don't dismiss short backtests outright."
            + _RETAIL_CONTEXT
        ),
        "focus": "Check for look-ahead bias, overfitting, parameter sensitivity, and result robustness.",
    },
    {
        "name": "Financial Analyst",
        "emoji": "💼",
        "system": (
            "You are a market analyst who helps retail traders understand when their strategy works. "
            "Explain clearly: does this work in bull markets only? Does it break in high-VIX periods? "
            "What macro conditions are friends or enemies of this strategy? "
            "Give specific, actionable regime filters a retail trader can actually monitor."
            + _RETAIL_CONTEXT
        ),
        "focus": "Explain regime dependency in plain terms — when to run this strategy and when to pause it.",
    },
    {
        "name": "Python Developer",
        "emoji": "💻",
        "system": (
            "You are a Python developer specializing in retail algo-trading systems. "
            "You review code for: correctness, look-ahead bias bugs, yfinance data handling, "
            "index alignment errors, and NaN handling. You know the common pitfalls of backtesting "
            "with pandas — off-by-one errors, using .shift() correctly, avoiding future leakage. "
            "Be specific about any bugs and give the exact fix."
            + _RETAIL_CONTEXT
        ),
        "focus": "Find code bugs, look-ahead bias, data handling errors — give specific line-level fixes.",
    },
]

_DEBATE_SYSTEM = """\
You are a senior trading mentor synthesizing feedback from 5 reviewers for a RETAIL algo-trader.
The trader has $5k–$100k, uses Python + yfinance, and trades on IBKR or Alpaca.

Calibrate your verdict for retail reality:
- TRADE LIVE: Sharpe > 0.35, profit factor > 1.2, max drawdown < 30%, no critical bugs, automatable
- PAPER TRADE: Has edge but needs refinement — Sharpe 0.2–0.35 OR one fixable concern
- NEEDS WORK: Real promise but has a specific issue that must be resolved first
- REJECT: Look-ahead bias confirmed, or strategy fundamentally broken

Return JSON:
{
  "consensus": "one clear paragraph verdict in plain language a retail trader understands",
  "confidence_score": 7,
  "confidence_reasoning": "why this score (1-10, where 10 = run this live today)",
  "key_strengths": ["top 3 practical strengths"],
  "key_concerns": ["top 3 real concerns — only include genuine problems"],
  "best_market_conditions": "plain English — when to run this",
  "avoid_conditions": "plain English — when to pause or stop",
  "verdict": "TRADE LIVE|PAPER TRADE|NEEDS WORK|REJECT",
  "next_steps": ["3 specific, implementable improvements a retail trader can actually do"],
  "retail_score": {
    "automation_ease": "1-10 — how easy to automate",
    "min_capital": "estimated minimum USD to trade this",
    "data_cost": "Free|Cheap|Expensive",
    "time_commitment": "5 min/week|Daily check|Active monitoring"
  }
}
Return only valid JSON."""


def _agent_context(parsed: dict, metrics: dict, regimes: dict,
                   code: str, prev: list[dict], issues: dict) -> str:
    lines = [
        "TRADER PROFILE: Retail algo-trader, $5k–$100k account, Python + yfinance, IBKR/Alpaca broker.",
        "EVALUATION STANDARD: A Sharpe > 0.3 and profit factor > 1.2 is tradeable for retail.",
        "",
        f"STRATEGY: {parsed.get('strategy_name', 'Unknown')}",
        f"Description: {parsed.get('description', '')}",
        f"Entry: {parsed.get('entry_logic', '')}",
        f"Exit: {parsed.get('exit_logic', '')}",
        f"Risk Mgmt: {parsed.get('risk_management', '')}",
        "",
        "BACKTEST METRICS:",
        f"  CAGR: {metrics.get('cagr_pct', 0):.2f}%",
        f"  Sharpe: {metrics.get('sharpe', 0):.3f}",
        f"  Sortino: {metrics.get('sortino', 0):.3f}",
        f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"  Calmar: {metrics.get('calmar', 0):.3f}",
        f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%",
        f"  Profit Factor: {metrics.get('profit_factor', 0):.3f}",
        f"  Expectancy: ${metrics.get('expectancy_usd', 0):.2f}",
        f"  Total Trades: {metrics.get('total_trades', 0)}",
        f"  Years Tested: {metrics.get('years', 0):.1f}",
        f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%",
        "",
        "MARKET REGIMES (SPY):",
        f"  Bull: {regimes.get('bull_pct', 0):.1f}%  Bear: {regimes.get('bear_pct', 0):.1f}%",
        f"  Annualized Vol: {regimes.get('annualized_vol', 0):.2f}%",
        f"  SPY Return same period: {regimes.get('spy_total_return', 0):.2f}%",
        "",
        "ISSUES DETECTED:",
    ]
    for iss in issues.get("issues", [])[:5]:
        lines.append(f"  [{iss.get('severity','?')}] {iss.get('type','?')}: {iss.get('description','')}")
    lines.append(f"  Code quality: {issues.get('overall_quality', 'Unknown')}")
    lines.append("")
    lines.append("STRATEGY CODE:")
    lines.append(f"```python\n{code}\n```")
    if prev:
        lines.append("\nPREVIOUS AGENT ANALYSES:")
        for a in prev:
            lines.append(f"\n[{a['name']}]:\n{a['analysis']}")
    return "\n".join(lines)


def run_agent_debate(parsed: dict, metrics: dict, regimes: dict,
                     code: str, issues: dict, jid: str) -> dict:
    """Run 5 sequential agents then synthesize into a debate consensus."""
    analyses = []

    for i, ag in enumerate(_AGENTS):
        _upd(jid,
             step=f"Agent {i+1}/5: {ag['name']} analyzing...",
             progress=35 + i * 8,
             **{"log+": f"{ag['emoji']} {ag['name']} analyzing"})

        ctx  = _agent_context(parsed, metrics, regimes, code, analyses, issues)
        text = _grok(
            ag["system"],
            f"{ag['focus']}\n\n{ctx}",
            max_tokens=700, jid=jid,
        )
        analyses.append({"name": ag["name"], "emoji": ag["emoji"], "analysis": text})
        _upd(jid, **{"log+": f"{ag['emoji']} {ag['name']} done"})

    # Debate synthesis
    _upd(jid, step="Synthesizing agent debate...", progress=76,
         **{"log+": "[DeepSeek] CIO synthesizing team debate"})
    team_summary = "\n\n".join(
        f"[{a['name']}]:\n{a['analysis']}" for a in analyses
    )
    raw = _grok(_DEBATE_SYSTEM,
                f"Synthesize these expert analyses:\n\n{team_summary}\n\nMetrics: "
                f"CAGR={metrics.get('cagr_pct',0):.2f}%, "
                f"Sharpe={metrics.get('sharpe',0):.3f}, "
                f"Max DD={metrics.get('max_drawdown_pct',0):.2f}%",
                max_tokens=1200, jid=jid)
    try:
        consensus = _extract_json(raw)
    except Exception:
        consensus = {"consensus": raw, "confidence_score": 5, "verdict": "NEEDS WORK"}

    return {"agents": analyses, "consensus": consensus}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SELF-IMPROVING EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_EVOLVE_SYSTEM = """\
You are a senior quant developer improving a trading strategy.
You have the original code, backtest results, and expert team feedback.

CRITICAL RULES:
- Remove any look-ahead bias
- Do NOT overfit to past data — prefer simpler, more robust signals
- Focus on: robustness over max returns, stability over optimization
- Improve exits and risk management first, then entry signals
- Reject changes that reduce statistical validity

Return JSON:
{
  "version": "v2.0",
  "changes": ["specific change 1", "specific change 2"],
  "reasoning": "why these changes improve robustness",
  "statistical_validity": "why this version is not overfit",
  "signal_generator": "complete improved generate_signals(df) function"
}
Return only valid JSON."""


def evolve_strategy(signal_code: str, analyses: list[dict], consensus: dict,
                    metrics: dict, version: int, jid: str) -> dict:
    _upd(jid, step=f"Evolving to v{version}...", **{"log+": f"Starting evolution v{version}"})

    feedback = "\n\n".join(f"[{a['name']}]:\n{a['analysis']}" for a in analyses)
    next_steps = "\n".join(f"- {s}" for s in consensus.get("next_steps", []))

    prompt = (
        f"Produce version v{version}.0 of this strategy.\n\n"
        f"CURRENT METRICS: CAGR={metrics.get('cagr_pct',0):.2f}%, "
        f"Sharpe={metrics.get('sharpe',0):.3f}, "
        f"Max DD={metrics.get('max_drawdown_pct',0):.2f}%, "
        f"Win Rate={metrics.get('win_rate_pct',0):.2f}%\n\n"
        f"TEAM RECOMMENDED NEXT STEPS:\n{next_steps}\n\n"
        f"ORIGINAL CODE:\n```python\n{signal_code}\n```\n\n"
        f"AGENT FEEDBACK:\n{feedback}\n\n"
        "Return only valid JSON."
    )
    # Grok produces the improved strategy concept; Haiku extracts the signal code
    evo_raw = _grok(_EVOLVE_SYSTEM, prompt, max_tokens=2000, jid=jid)
    evo = _extract_json(evo_raw)

    # Re-generate signal code with Haiku for clean, executable output
    if evo.get("signal_generator", "").strip():
        code_prompt = (
            f"Clean up and complete this generate_signals(df) function. "
            f"Return ONLY the function, no markdown, no explanation:\n\n"
            f"{evo['signal_generator']}"
        )
        clean_code = _haiku(_CODEGEN_SYSTEM, code_prompt, max_tokens=1500, jid=jid)
        evo["signal_generator"] = clean_code
    return evo


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

_REPORT_SYSTEM = """\
You are a trading mentor writing a practical evaluation report for a RETAIL algo-trader.
Write a clear, jargon-free summary (max 600 words) covering:
1. VERDICT — trade live / paper trade / needs work / reject — and WHY in 1 sentence
2. THE EDGE — what is this strategy actually exploiting? Does the logic make sense?
3. BACKTEST REALITY CHECK — what do the numbers actually mean for a retail account?
   (e.g. "A 0.45 Sharpe means roughly 1 good year for every 2 — that's tradeable")
4. WHAT WORKS — top 2-3 genuine strengths
5. WATCH OUT FOR — top 2-3 real risks (skip theoretical hedge-fund concerns)
6. EVOLUTION — did the AI improvements help? What changed?
7. HOW TO RUN IT — broker, minimum capital, how often to check it, data needed
8. CONFIDENCE — X/10 and what would move the needle

Write like you're advising a friend, not filing an SEC report. Be honest but encouraging."""


def generate_final_report(parsed: dict, v1_metrics: dict, evolution: list[dict],
                           debate: dict, jid: str) -> str:
    _upd(jid, step="Writing final report...", **{"log+": "CIO writing final report"})
    best_ver = max(
        [{"version": "v1", "metrics": v1_metrics}] +
        [{"version": f"v{e['version']}", "metrics": e["metrics"]} for e in evolution],
        key=lambda x: x["metrics"].get("sharpe", 0),
    )
    evo_summary = "\n".join(
        f"v{e['version']}: CAGR={e['metrics'].get('cagr_pct',0):.2f}%, "
        f"Sharpe={e['metrics'].get('sharpe',0):.3f}, "
        f"Max DD={e['metrics'].get('max_drawdown_pct',0):.2f}%  "
        f"Changes: {'; '.join(e.get('changes', [])[:2])}"
        for e in evolution
    )
    consensus = debate.get("consensus", {})
    prompt = (
        f"Strategy: {parsed.get('strategy_name', 'Unknown')}\n"
        f"Description: {parsed.get('description', '')}\n\n"
        f"v1 Metrics: CAGR={v1_metrics.get('cagr_pct',0):.2f}%, "
        f"Sharpe={v1_metrics.get('sharpe',0):.3f}, "
        f"Max DD={v1_metrics.get('max_drawdown_pct',0):.2f}%, "
        f"Win Rate={v1_metrics.get('win_rate_pct',0):.2f}%\n\n"
        f"Evolution:\n{evo_summary or 'No evolution ran'}\n"
        f"Best version: {best_ver['version']} (Sharpe {best_ver['metrics'].get('sharpe',0):.3f})\n\n"
        f"Team verdict: {consensus.get('verdict', '—')}\n"
        f"Confidence: {consensus.get('confidence_score', '—')}/10\n"
        f"Key concerns: {'; '.join(consensus.get('key_concerns', []))}\n\n"
        "Write the final executive report now."
    )
    return _grok(_REPORT_SYSTEM, prompt, max_tokens=1400, jid=jid)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORCHESTRATORS
# ═══════════════════════════════════════════════════════════════════════════════

_RUNTIME_ERROR_KEYWORDS = [
    "do not exist", "KeyError", "AttributeError", "TypeError",
    "not in index", "generate_signals", "IndexError", "ValueError",
    "signal", "column",
]


def _run_backtest_pipeline(signal_code: str, parsed: dict, config: dict, jid: str) -> dict:
    """Common backtest + metrics + regime analysis step."""
    symbols  = config["symbols"]
    start    = config["start"]
    end      = config["end"]
    capital  = config["capital"]
    interval = config["interval"]

    _upd(jid, step="Running backtest...", **{"log+": f"Backtesting {len(symbols)} symbols ({interval})"})

    # Build signal function with compile-error auto-fix loop
    signal_fn, current_code = _build_signal_fn_with_fix(signal_code, jid)
    engine = BacktestEngine(symbols, start, end, capital,
                            config["slippage"], config["commission"],
                            config.get("stop_loss"), interval)
    result = engine.run(signal_fn)

    # If runtime error looks signal-related, attempt up to 2 more Haiku fixes
    if "error" in result:
        err_msg = result["error"]
        if any(kw in err_msg for kw in _RUNTIME_ERROR_KEYWORDS):
            for rt_attempt in range(1, 3):
                _upd(jid, **{"log+": f"[Haiku] Runtime error (attempt {rt_attempt}): {err_msg[:120]} — fixing..."})
                try:
                    fixed_code = _autofix_signal_code(current_code, err_msg, rt_attempt, jid)
                    signal_fn2, fixed_code = _build_signal_fn_with_fix(fixed_code, jid, max_attempts=2)
                    result2 = engine.run(signal_fn2)
                    if "error" not in result2:
                        result = result2
                        current_code = fixed_code
                        _upd(jid, **{"log+": f"[Haiku] Runtime fix succeeded on attempt {rt_attempt}"})
                        break
                    err_msg = result2["error"]
                except Exception as exc:
                    err_msg = str(exc)

    if "error" in result:
        raise RuntimeError(result["error"])

    for fe in result.get("fetch_errors", []):
        _upd(jid, **{"log+": f"Fetch warning: {fe}"})

    equity  = result["equity"]
    trades  = result["trades"]
    metrics = compute_metrics(equity, trades, capital)
    _upd(jid, **{"log+": f"CAGR={metrics.get('cagr_pct',0):.2f}% Sharpe={metrics.get('sharpe',0):.3f}"})
    return {
        "metrics":       metrics,
        "equity_points": _equity_points(equity),
        "signal_code":   current_code,   # may be auto-fixed version
        "trades":        trades,
        "regime_breakdown": regime_breakdown(trades, start, end),
    }


def _run_evolution(signal_code: str, analyses: list[dict], consensus: dict,
                   metrics: dict, config: dict, jid: str) -> list[dict]:
    """Run 3 evolution generations, keep best version."""
    history   = []
    best      = {"sharpe": metrics.get("sharpe", 0), "code": signal_code, "metrics": metrics}
    cur_code  = signal_code
    cur_metr  = metrics

    for gen in range(2, 5):   # v2, v3, v4
        _upd(jid, progress=80 + (gen - 2) * 4)
        try:
            evo      = evolve_strategy(cur_code, analyses, consensus, cur_metr, gen, jid)
            new_code = evo.get("signal_generator", "")
            if not new_code.strip():
                _upd(jid, **{"log+": f"v{gen}: empty code — skipping"}); continue

            bt = _run_backtest_pipeline(new_code, {}, config, jid)
            new_metr = bt["metrics"]

            history.append({
                "version":      gen,
                "changes":      evo.get("changes", []),
                "reasoning":    evo.get("reasoning", ""),
                "metrics":      new_metr,
                "equity_points": bt["equity_points"],
                "signal_code":  new_code,
                "improved":     new_metr.get("sharpe", 0) > cur_metr.get("sharpe", 0),
            })
            _upd(jid, **{"log+": f"v{gen}: CAGR={new_metr.get('cagr_pct',0):.2f}% "
                                   f"Sharpe={new_metr.get('sharpe',0):.3f} "
                                   f"({'improved' if history[-1]['improved'] else 'no improvement'})"})

            if new_metr.get("sharpe", 0) > best["sharpe"]:
                best = {"sharpe": new_metr["sharpe"], "code": new_code, "metrics": new_metr}
                cur_code = new_code
                cur_metr = new_metr
            else:
                _upd(jid, **{"log+": f"v{gen}: rejected — Sharpe did not improve"})

        except Exception as exc:
            _upd(jid, **{"log+": f"v{gen} evolution error: {exc}"})

    return history


def run_validation_job(jid: str, code: str, config: dict) -> None:
    """Workflow 1: Validate and evolve an existing strategy."""
    try:
        claude_label = "Sonnet 4.6" if "sonnet" in _active_claude_model["model"] else "Haiku 4.5"
        _upd(jid, status="running", progress=5,
             **{"log+": f"Validation job started | Claude model: {claude_label}"})

        # Parse strategy
        _upd(jid, progress=8)
        parsed = parse_strategy(code, jid)
        _upd(jid, progress=14, **{"log+": f"Parsed: {parsed.get('strategy_name')}"})

        # Override config from code if not user-specified
        if not config.get("symbols") and parsed.get("symbols"):
            config["symbols"] = [s.upper() for s in parsed["symbols"] if s]
            _upd(jid, **{"log+": f"Symbols from code: {config['symbols']}"})
        if parsed.get("start_date") and _valid_date(parsed["start_date"]) and not config.get("_user_dates"):
            config["start"] = parsed["start_date"]
        if parsed.get("end_date") and _valid_date(parsed["end_date"]) and not config.get("_user_dates"):
            config["end"] = parsed["end_date"]
        if parsed.get("interval"):
            config["interval"] = _norm_interval(parsed["interval"])
        _upd(jid, **{"log+": f"Interval: {config['interval']} | {config['start']} → {config['end']}"})

        # Issue detection
        _upd(jid, progress=16)
        issues = detect_issues(code, jid)
        issue_count = len(issues.get("issues", []))
        _upd(jid, **{"log+": f"Issues found: {issue_count} ({issues.get('overall_quality','?')} quality)"})

        # Detect regimes
        _upd(jid, step="Analysing market regimes...", progress=20,
             **{"log+": "Fetching SPY regime data"})
        regimes = detect_regimes(config["start"], config["end"])

        # v1 backtest
        _upd(jid, progress=24)
        signal_code = parsed.get("signal_generator", "")
        v1 = _run_backtest_pipeline(signal_code, parsed, config, jid)
        _upd(jid, progress=32)

        # Multi-agent debate
        debate = run_agent_debate(parsed, v1["metrics"], regimes, code, issues, jid)
        _upd(jid, progress=77)

        # Evolution — use auto-fixed code if the fix loop changed it
        evolution = _run_evolution(
            v1["signal_code"], debate["agents"], debate["consensus"],
            v1["metrics"], config, jid,
        )
        _upd(jid, progress=92)

        # Final report
        final_report = generate_final_report(parsed, v1["metrics"], evolution, debate, jid)

        # Best version
        all_versions = [{"version": 1, "metrics": v1["metrics"], "sharpe": v1["metrics"].get("sharpe", 0)}]
        for e in evolution:
            all_versions.append({"version": e["version"], "metrics": e["metrics"], "sharpe": e["metrics"].get("sharpe", 0)})
        best_version = max(all_versions, key=lambda x: x["sharpe"])

        tok = _token_cost(jid)
        _claude_key = _claude_model_key()
        _claude_tok = tok.get(_claude_key, {})
        _upd(jid, **{"log+": f"Done — est. cost: ${tok.get('total_cost_usd',0):.4f} "
                              f"(DeepSeek {tok.get('grok',{}).get('in',0)+tok.get('grok',{}).get('out',0):,} tokens, "
                              f"{_claude_key.capitalize()} {_claude_tok.get('in',0)+_claude_tok.get('out',0):,} tokens)"})
        _upd(jid,
             status="done", progress=100, step="Complete",
             result={
                 "mode":           "validate",
                 "parsed":         parsed,
                 "issues":         issues,
                 "regimes":        regimes,
                 "v1":             {k: v for k, v in v1.items() if k != "trades"},
                 "agents":         debate["agents"],
                 "consensus":      debate["consensus"],
                 "evolution":      evolution,
                 "final_report":   final_report,
                 "best_version":   best_version["version"],
                 "token_usage":    tok,
                 "config":         {
                     "symbols":        config["symbols"],
                     "start":          config["start"],
                     "end":            config["end"],
                     "capital":        config["capital"],
                     "interval":       config["interval"],
                     "interval_label": _INTERVAL_LABEL.get(config["interval"], config["interval"]),
                 },
             },
             **{"log+": "Complete"})

    except Exception as exc:
        _upd(jid, status="error", error=str(exc),
             **{"log+": f"ERROR: {exc}\n{traceback.format_exc()}"})


def run_generation_job(jid: str, idea: str, config: dict) -> None:
    """Workflow 2: Research idea → generate code → backtest → evolve."""
    try:
        claude_label = "Sonnet 4.6" if "sonnet" in _active_claude_model["model"] else "Haiku 4.5"
        _upd(jid, status="running", progress=5,
             **{"log+": f"Generation job started | Claude model: {claude_label}"})

        # Research the idea
        concept = research_idea(idea, jid)
        _upd(jid, progress=15, **{"log+": f"Concept: {concept.get('strategy_name')}"})

        # Use concept symbols if user didn't specify
        if not config.get("symbols") and concept.get("universe"):
            config["symbols"] = [s.upper() for s in concept["universe"][:5] if s]
            _upd(jid, **{"log+": f"Symbols: {config['symbols']}"})

        # Generate signal code
        signal_code = generate_strategy_code(concept, jid)
        _upd(jid, progress=25, **{"log+": "Strategy code generated"})

        # Build a parsed dict from the concept (no code parsing needed)
        parsed = {
            "strategy_name": concept.get("strategy_name", "Generated Strategy"),
            "description":   concept.get("rationale", ""),
            "indicators":    concept.get("indicators", []),
            "entry_logic":   concept.get("entry_rules", ""),
            "exit_logic":    concept.get("exit_rules", ""),
            "risk_management": concept.get("risk_management", ""),
            "timeframe":     concept.get("timeframe", "daily"),
            "signal_generator": signal_code,
        }

        # Run issue detection on generated code
        issues = detect_issues(signal_code, jid)
        _upd(jid, **{"log+": f"Issues: {len(issues.get('issues',[]))} found"})

        # Regimes
        _upd(jid, step="Analysing market regimes...", progress=28,
             **{"log+": "Fetching SPY"})
        regimes = detect_regimes(config["start"], config["end"])

        # v1 backtest
        v1 = _run_backtest_pipeline(signal_code, parsed, config, jid)
        _upd(jid, progress=40)

        # Multi-agent debate — use auto-fixed code from v1 if it changed
        debate = run_agent_debate(parsed, v1["metrics"], regimes, v1["signal_code"], issues, jid)
        _upd(jid, progress=77)

        # Evolution — use auto-fixed code if the fix loop changed it
        evolution = _run_evolution(
            v1["signal_code"], debate["agents"], debate["consensus"],
            v1["metrics"], config, jid,
        )
        _upd(jid, progress=92)

        final_report = generate_final_report(parsed, v1["metrics"], evolution, debate, jid)

        all_versions = [{"version": 1, "metrics": v1["metrics"], "sharpe": v1["metrics"].get("sharpe", 0)}]
        for e in evolution:
            all_versions.append({"version": e["version"], "metrics": e["metrics"], "sharpe": e["metrics"].get("sharpe", 0)})
        best_version = max(all_versions, key=lambda x: x["sharpe"])

        tok = _token_cost(jid)
        _claude_key = _claude_model_key()
        _claude_tok = tok.get(_claude_key, {})
        _upd(jid, **{"log+": f"Done — est. cost: ${tok.get('total_cost_usd',0):.4f} "
                              f"(DeepSeek {tok.get('grok',{}).get('in',0)+tok.get('grok',{}).get('out',0):,} tokens, "
                              f"{_claude_key.capitalize()} {_claude_tok.get('in',0)+_claude_tok.get('out',0):,} tokens)"})
        _upd(jid,
             status="done", progress=100, step="Complete",
             result={
                 "mode":          "generate",
                 "idea":          idea,
                 "concept":       concept,
                 "parsed":        parsed,
                 "issues":        issues,
                 "regimes":       regimes,
                 "v1":            {k: v for k, v in v1.items() if k != "trades"},
                 "agents":        debate["agents"],
                 "consensus":     debate["consensus"],
                 "evolution":     evolution,
                 "final_report":  final_report,
                 "best_version":  best_version["version"],
                 "token_usage":   tok,
                 "config": {
                     "symbols":        config["symbols"],
                     "start":          config["start"],
                     "end":            config["end"],
                     "capital":        config["capital"],
                     "interval":       config["interval"],
                     "interval_label": _INTERVAL_LABEL.get(config["interval"], config["interval"]),
                 },
             },
             **{"log+": "Complete"})

    except Exception as exc:
        _upd(jid, status="error", error=str(exc),
             **{"log+": f"ERROR: {exc}\n{traceback.format_exc()}"})


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def start_lab_job(payload: dict) -> str:
    mode  = payload.get("mode", "validate")
    conf  = payload.get("config") or {}
    now   = datetime.now()

    # Set active Claude model for code generation
    requested_model = conf.pop("claude_model", _HAIKU_MODEL)
    if requested_model in (_HAIKU_MODEL, _SONNET_MODEL):
        _active_claude_model["model"] = requested_model
    else:
        _active_claude_model["model"] = _HAIKU_MODEL

    # Resolve config defaults
    conf.setdefault("symbols",    ["SPY", "QQQ", "AAPL"])
    conf.setdefault("end",        now.strftime("%Y-%m-%d"))
    conf.setdefault("start",      (now - timedelta(days=365 * 3)).strftime("%Y-%m-%d"))
    conf.setdefault("capital",    100_000.0)
    conf.setdefault("slippage",   0.0005)
    conf.setdefault("commission", 0.005)
    conf.setdefault("stop_loss",  None)
    conf["interval"] = _norm_interval(conf.get("interval", "1d"))

    # Track whether user explicitly set dates
    if conf.get("start") != (now - timedelta(days=365 * 3)).strftime("%Y-%m-%d"):
        conf["_user_dates"] = True

    jid = _new_job(mode)
    if mode == "generate":
        idea = payload.get("idea", "")
        t = threading.Thread(target=run_generation_job, args=(jid, idea, conf), daemon=True)
    else:
        code = payload.get("code", "")
        t = threading.Thread(target=run_validation_job, args=(jid, code, conf), daemon=True)
    t.start()
    return jid
