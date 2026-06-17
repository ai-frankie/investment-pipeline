"""
quiver_congress_watchlist.py
Pulls congressional stock trades (House + Senate) from free public endpoints.
No API key required. Data sourced from STOCK Act filings.

Sources:
    House: https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json
    Senate: https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com/aggregate/all_transactions.json

Usage:
    python quiver_congress_watchlist.py
    python quiver_congress_watchlist.py --days 30
    python quiver_congress_watchlist.py --days 7 --show-all
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

WATCHLIST_PATH = "watchlist.txt"
OUTPUT_DIR = Path("output/congress")

HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"


def load_watchlist(path: str) -> set:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")
    with open(p) as f:
        return {line.strip().upper() for line in f if line.strip()}


def fetch_house() -> pd.DataFrame:
    print("Fetching House trades...")
    resp = requests.get(HOUSE_URL, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["chamber"] = "House"
    return df


def fetch_senate() -> pd.DataFrame:
    print("Fetching Senate trades...")
    resp = requests.get(SENATE_URL, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["chamber"] = "Senate"
    return df


def normalize(df: pd.DataFrame, chamber: str) -> pd.DataFrame:
    """Normalize House or Senate DataFrame to common schema."""
    col_map = {}

    if chamber == "House":
        col_map = {
            "ticker": "ticker",
            "representative": "politician",
            "type": "side",
            "transaction_date": "date",
            "amount": "amount_range",
            "chamber": "chamber",
        }
    elif chamber == "Senate":
        col_map = {
            "ticker": "ticker",
            "senator": "politician",
            "type": "side",
            "transaction_date": "date",
            "amount": "amount_range",
            "chamber": "chamber",
        }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Keep only columns we care about
    keep = [c for c in ["ticker", "politician", "side", "date", "amount_range", "chamber"] if c in df.columns]
    df = df[keep].copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"])

    # Drop non-stock rows (e.g. "--", "N/A")
    df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")]

    return df


def get_signals(tickers, days: int = 30) -> dict:
    """
    Quiet helper for pipeline.py: returns {ticker: {"buys": n, "sells": n}}
    for recent congressional trades. Full House+Senate dump is cached to disk
    once per day (the source JSON files are large).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    cache = OUTPUT_DIR / f"all_trades_{today}.csv"

    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
    else:
        df = None
        # Primary: Quiver API (free key in QUIVER_API_KEY env var) — the old
        # stock-watcher S3 buckets now return 403 (project abandoned).
        import os
        api_key = os.getenv("QUIVER_API_KEY")
        if api_key:
            try:
                import quiverquant
                q = quiverquant.quiver(api_key)
                raw = q.congress_trading()
                raw = raw.rename(columns={
                    "Ticker": "ticker", "Representative": "politician",
                    "Transaction": "side", "Date": "date", "Amount": "amount_range",
                })
                raw["ticker"] = raw["ticker"].astype(str).str.upper().str.strip()
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
                df = raw.dropna(subset=["date", "ticker"])
            except Exception as e:
                print(f"[CONGRESS] Quiver fetch failed: {e}")
        if df is None:
            frames = []
            for fetcher, chamber in ((fetch_house, "House"), (fetch_senate, "Senate")):
                try:
                    frames.append(normalize(fetcher(), chamber))
                except Exception as e:
                    print(f"[CONGRESS] {chamber} fetch failed: {e}")
            if not frames:
                return {}
            df = pd.concat(frames, ignore_index=True)
        df.to_csv(cache, index=False)

    cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[df["date"] >= cutoff]

    out = {}
    for t in tickers:
        sub = df[df["ticker"] == t.upper()]
        if sub.empty:
            continue
        side = sub["side"].astype(str).str.lower()
        buys = int(side.str.contains("purchase|buy", regex=True).sum())
        sells = int(side.str.contains("sale|sell", regex=True).sum())
        out[t] = {"buys": buys, "sells": sells}
    return out


def run(days: int = 7, watchlist_path: str = WATCHLIST_PATH, show_all: bool = False):
    watchlist = load_watchlist(watchlist_path)
    print(f"Watchlist: {sorted(watchlist)}")

    # Fetch both chambers
    frames = []
    try:
        frames.append(normalize(fetch_house(), "House"))
    except Exception as e:
        print(f"House fetch failed: {e}")

    try:
        frames.append(normalize(fetch_senate(), "Senate"))
    except Exception as e:
        print(f"Senate fetch failed: {e}")

    if not frames:
        print("No data fetched from either chamber.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Filter by date window
    cutoff = datetime.utcnow() - timedelta(days=days)
    df_recent = df[df["date"] >= cutoff].copy()

    print(f"\nTotal trades in last {days} days: {len(df_recent)}")

    # Filter to watchlist
    df_hits = df_recent[df_recent["ticker"].isin(watchlist)].sort_values("date", ascending=False)

    print("\n" + "=" * 60)
    print(f"Watchlist hits (last {days} days):")
    print("=" * 60)
    if df_hits.empty:
        print("None found. Try --days 30 or --days 90.")
    else:
        print(df_hits.to_string(index=False))

    # Top 20 most traded tickers by politicians
    print(f"\nTop 20 most politically traded tickers (last {days} days):")
    top = df_recent["ticker"].value_counts().head(20)
    print(top.to_string())

    # Buys only — strongest signal
    buys = df_recent[df_recent["side"].str.contains("Purchase|Buy|purchase|buy", na=False)]
    print(f"\nTop 10 BUY targets by politicians (last {days} days):")
    print(buys["ticker"].value_counts().head(10).to_string())

    # Save snapshot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"congress_watchlist_{today}.csv"
    df_hits.to_csv(out_path, index=False)

    # Also save full top-buys for watchlist expansion ideas
    top_buys_path = OUTPUT_DIR / f"congress_top_buys_{today}.csv"
    buys["ticker"].value_counts().head(50).to_frame("trade_count").to_csv(top_buys_path)

    print(f"\nSaved watchlist hits -> {out_path}")
    print(f"Saved top buys list  -> {top_buys_path}")

    if show_all:
        print("\nAll recent trades:")
        print(df_recent.sort_values("date", ascending=False).head(50).to_string(index=False))

    return df_hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Congressional trade watchlist filter (free, no API key)")
    parser.add_argument("--days", default=7, type=int, help="Lookback window in days (try 30 or 90)")
    parser.add_argument("--watchlist", default=WATCHLIST_PATH, help="Path to watchlist.txt")
    parser.add_argument("--show-all", action="store_true", help="Print all recent trades (top 50)")
    args = parser.parse_args()
    run(days=args.days, watchlist_path=args.watchlist, show_all=args.show_all)
