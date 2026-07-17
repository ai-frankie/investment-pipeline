"""
robinhood_fetcher.py
Drop-in replacement for kronos_data_fetcher.py — same fetch_ohlcv() signature.

Priority chain per situation:
  Daily / hourly  : Robinhood (real-time) → yfinance (15min delay) → error
  Intraday 5m     : Robinhood            → yfinance               → error
  Intraday 1m/15m/30m: yfinance only (RH doesn't support these)
  Future intraday : Fidelity → Robinhood → yfinance  (see FIDELITY stub below)

Auth (Robinhood): stored session token if present, else RH_USERNAME / RH_PASSWORD
from .env in this directory.

Usage:
    python robinhood_fetcher.py --ticker AAPL --interval 1h --lookback 400
"""

import argparse
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Interval config
# ---------------------------------------------------------------------------

# Intervals that require real-time tick resolution for meaningful signals.
# Flip INTRADAY_MODE to True when ready to trade intraday — this promotes
# Fidelity to first in the chain (once _fetch_fidelity is implemented).
INTRADAY_MODE = False
INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m"}

# robin_stocks (interval, span) — None = not supported by RH
_RH_MAP = {
    "1m":  None,
    "5m":  ("5minute", "3month"),
    "15m": None,
    "30m": None,
    "1h":  ("hour",    "year"),
    "1d":  ("day",     "5year"),
}

# yfinance lookback periods
_YF_PERIOD = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "max",
}

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_rh_logged_in = False


def _ensure_rh_login() -> None:
    global _rh_logged_in
    if _rh_logged_in:
        return
    import robin_stocks.robinhood as rh
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    rh.login(
        username=os.environ.get("RH_USERNAME"),
        password=os.environ.get("RH_PASSWORD"),
        store_session=True,
    )
    _rh_logged_in = True


# ---------------------------------------------------------------------------
# Core fetch — priority chain
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    ticker: str,
    interval: str = "1h",
    lookback: int = 400,
    asof=None,
) -> pd.DataFrame:
    """
    Fetch OHLCV with automatic source failover.
    Returns DataFrame: timestamps, open, high, low, close, volume.
    Logs [RH], [FID], or [yf] prefix so you can see which source was used.
    """
    is_intraday = interval in INTRADAY_INTERVALS

    # Intraday mode: try Fidelity first (superior tick data), then fall through
    if is_intraday and INTRADAY_MODE:
        try:
            return _fetch_fidelity(ticker, interval, lookback, asof)
        except NotImplementedError:
            print("[FID] not yet implemented — continuing to RH/yf")
        except Exception as e:
            print(f"[FID] failed ({e}) — continuing to RH/yf")

    # Robinhood (real-time, supported intervals only)
    rh_params = _RH_MAP.get(interval)
    if rh_params is not None:
        try:
            return _fetch_robinhood(ticker, interval, rh_params, lookback, asof)
        except Exception as e:
            print(f"[RH] failed ({e}) — falling back to yfinance")

    # yfinance (15min delay — fine for daily/hourly; last resort for intraday)
    return _fetch_yfinance(ticker, interval, lookback, asof)


# ---------------------------------------------------------------------------
# Source implementations
# ---------------------------------------------------------------------------
def _fetch_robinhood(ticker, yf_interval, rh_params, lookback, asof):
    import robin_stocks.robinhood as rh
    rh_interval, rh_span = rh_params
    print(f"[RH] {ticker} @ {yf_interval} (span={rh_span})")
    _ensure_rh_login()

    raw = rh.stocks.get_stock_historicals(
        ticker, interval=rh_interval, span=rh_span, bounds="regular"
    )
    if not raw:
        raise ValueError(f"Empty response from Robinhood for {ticker}")

    df = pd.DataFrame(raw)[
        ["begins_at", "open_price", "high_price", "low_price", "close_price", "volume"]
    ].copy()
    df.columns = ["timestamps", "open", "high", "low", "close", "volume"]
    # RH's begins_at is UTC ISO8601 — convert to Eastern, then strip tz to
    # naive-Eastern (was: strip UTC tz directly, silently mislabeling UTC
    # clock time as Eastern -> every candle off by the UTC/ET offset).
    df["timestamps"] = (pd.to_datetime(df["timestamps"], utc=True)
                         .dt.tz_convert("America/New_York").dt.tz_localize(None))
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamps").reset_index(drop=True)
    return _trim(df, ticker, lookback, asof)


def _fetch_yfinance(ticker, interval, lookback, asof):
    import yfinance as yf
    period = _YF_PERIOD.get(interval, "730d")
    print(f"[yf]  {ticker} @ {interval} (period={period})")

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data from yfinance for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamps"
    df = df.reset_index()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.dropna().reset_index(drop=True)
    return _trim(df, ticker, lookback, asof)


def _fetch_fidelity(ticker, interval, lookback, asof):
    # ---------------------------------------------------------------------------
    # FIDELITY STUB — activate when ready for intraday trading
    #
    # Options to implement:
    #   1. fidelity-api (unofficial, selenium-based): pip install fidelity
    #      https://github.com/enigmamachine/fidelity
    #   2. Fidelity ATP (Active Trader Pro) streaming — requires ATP desktop app
    #   3. Polygon.io API (paid, $29/mo) — best tick data, easiest API
    #      Set POLYGON_API_KEY in .env and use requests/polygon-api-client
    #
    # When implemented, set INTRADAY_MODE = True at top of this file.
    # ---------------------------------------------------------------------------
    raise NotImplementedError(
        "Fidelity fetcher not yet implemented. "
        "Set INTRADAY_MODE=False to skip. See stub above for options."
    )


# ---------------------------------------------------------------------------
# Shared trim
# ---------------------------------------------------------------------------
def _trim(df, ticker, lookback, asof):
    if asof is not None:
        cutoff = pd.to_datetime(asof)
        ts = df["timestamps"]
        if ts.dt.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(ts.dt.tz)
        elif ts.dt.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        df = df[ts <= cutoff].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No candles for {ticker} at or before {asof}")
    if len(df) > lookback:
        df = df.tail(lookback).reset_index(drop=True)
    print(f"  {len(df)} candles | {df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}")
    return df


# ---------------------------------------------------------------------------
# Save helper (matches kronos_data_fetcher.save)
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame, ticker: str, interval: str, out_dir: str = ".") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker.replace('/', '_')}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV for Kronos")
    parser.add_argument("--ticker",   default="AAPL")
    parser.add_argument("--interval", default="1h", choices=list(_RH_MAP))
    parser.add_argument("--lookback", default=400, type=int)
    parser.add_argument("--out_dir",  default="data")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, args.interval, args.lookback)
    save(df, args.ticker, args.interval, args.out_dir)
