"""
kronos_data_fetcher.py
Fetches OHLCV data from Yahoo Finance and formats it for KronosPredictor.

Usage:
    python kronos_data_fetcher.py --ticker AAPL --interval 1h --lookback 400
"""

import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path


INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "1d":  "1d",
}

# yfinance max lookback per interval
PERIOD_MAP = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "max",
}


def fetch_ohlcv(ticker: str, interval: str = "1h", lookback: int = 400, asof=None) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker and return a clean DataFrame
    with columns: timestamps, open, high, low, close, volume

    asof: optional timestamp — drop all candles after this point (for
    walk-forward backtesting without look-ahead).
    """
    period = PERIOD_MAP.get(interval, "730d")
    print(f"Downloading {ticker} @ {interval} (period={period})...")

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

    # Flatten multi-level columns if present (yfinance quirk for single ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamps"
    df = df.reset_index()
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    # Drop rows with NaN (pre-market gaps etc.)
    df = df.dropna().reset_index(drop=True)

    if asof is not None:
        cutoff = pd.to_datetime(asof)
        ts = df["timestamps"]
        if ts.dt.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(ts.dt.tz)
        df = df[ts <= cutoff].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No candles for {ticker} at or before {asof}")

    # Trim to requested lookback
    if len(df) > lookback:
        df = df.tail(lookback).reset_index(drop=True)

    print(f"  Got {len(df)} candles | {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
    return df


def save(df: pd.DataFrame, ticker: str, interval: str, out_dir: str = ".") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker.replace('/', '_')}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for Kronos")
    parser.add_argument("--ticker",   default="AAPL",  help="Yahoo Finance ticker (e.g. AAPL, BTC-USD, SPY)")
    parser.add_argument("--interval", default="1h",    choices=list(INTERVAL_MAP), help="Candle interval")
    parser.add_argument("--lookback", default=400,     type=int, help="Number of candles to keep (Kronos-small/base max=512)")
    parser.add_argument("--out_dir",  default="data",  help="Output directory for CSV files")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, args.interval, args.lookback)
    save(df, args.ticker, args.interval, args.out_dir)
