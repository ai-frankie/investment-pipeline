"""
usaspending_watcher.py
Pulls recent government contract awards for watchlist companies from USASpending.gov.
No API key required. Data sourced from FPDS (Federal Procurement Data System).
Contracts typically appear within 3-7 business days of award.

Usage:
    python usaspending_watcher.py
    python usaspending_watcher.py --days 30 --min_amount 1000000
    python usaspending_watcher.py --days 7 --min_amount 10000000
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

OUTPUT_DIR = Path("output/contracts")
CACHE_MAX_AGE_DAYS = 7

# Ticker -> company search terms (USASpending uses company names, not tickers)
TICKER_TO_COMPANY = {
    # Mega-cap tech (background signal — contract mods common)
    "AAPL":  ["Apple Inc", "Apple Computer"],
    "MSFT":  ["Microsoft Corporation", "Microsoft Corp", "GitHub Inc"],
    "AMZN":  ["Amazon", "Amazon Web Services", "AWS"],
    "NVDA":  ["Nvidia Corporation", "Nvidia Corp"],
    "META":  ["Meta Platforms", "Facebook"],
    "GOOGL": ["Google LLC", "Alphabet"],
    # Defense/AI gov contractors (HIGH signal — single contracts are material)
    "PLTR":  ["Palantir Technologies", "Palantir"],
    "CACI":  ["CACI International", "CACI Inc"],
    "SAIC":  ["Science Applications International", "SAIC"],
    "BAH":   ["Booz Allen Hamilton", "Booz Allen"],
    "LDOS":  ["Leidos", "Leidos Inc", "Leidos Holdings"],
    # Defense primes (HIGH signal — single awards are material)
    "RTX":   ["Raytheon", "RTX Corporation", "Collins Aerospace", "Pratt & Whitney"],
    "LMT":   ["Lockheed Martin"],
    "NOC":   ["Northrop Grumman"],
    "GD":    ["General Dynamics"],
    # ETFs — no contracts
    "SPY":   [],
    "QQQ":   [],
}

BASE_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"


def _post_with_retry(url: str, payload: dict, timeout: int = 15,
                      backoffs=(1, 2, 4)) -> requests.Response:
    """3 attempts, 1s/2s/4s backoff. Retries network errors and 5xx server
    errors only — a 4xx means the request itself is malformed and retrying
    won't help."""
    for attempt in range(len(backoffs)):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise
            if attempt == len(backoffs) - 1:
                raise
            time.sleep(backoffs[attempt])
        except requests.exceptions.RequestException:
            if attempt == len(backoffs) - 1:
                raise
            time.sleep(backoffs[attempt])


def fetch_contracts(company_names: list, start_date: str, end_date: str, min_amount: float,
                     errors: list | None = None) -> pd.DataFrame:
    """Fetch contract awards for a list of company name variants. Appends
    (name, error) to `errors` (if given) on failure so the caller can tell a
    total-fetch-failure apart from a legitimate zero-results day."""
    all_results = []

    for name in company_names:
        payload = {
            "filters": {
                "recipient_search_text": [name],
                "award_type_codes": ["A", "B", "C", "D"],  # Contracts only
                "time_period": [{"start_date": start_date, "end_date": end_date}],
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Award Amount",
                "Start Date",
                "Description",
                "Awarding Agency",
                "Awarding Sub Agency",
            ],
            "page": 1,
            "limit": 20,
            "sort": "Start Date",
            "order": "desc",
        }

        try:
            resp = _post_with_retry(BASE_URL, payload, timeout=15)
            data = resp.json()
            results = data.get("results", [])
            for r in results:
                if r.get("Award Amount", 0) >= min_amount:
                    # Tag as new award vs modification of existing contract
                    # USASpending resurfaces old contracts on modification — both are signals
                    award_date = r.get("Start Date", "")
                    r["signal_type"] = "NEW AWARD" if award_date and award_date >= start_date else "MODIFICATION"
                    all_results.append(r)
        except Exception as e:
            print(f"  Error fetching '{name}': {e}")
            if errors is not None:
                errors.append((name, str(e)))

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df = df.rename(columns={
        "Award ID": "award_id",
        "Recipient Name": "company",
        "Award Amount": "amount",
        "Start Date": "award_date",
        "Description": "description",
        "Awarding Agency": "agency",
        "Awarding Sub Agency": "sub_agency",
    })
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["award_date"] = pd.to_datetime(df["award_date"], errors="coerce")
    return df.drop_duplicates(subset=["award_id"]).sort_values(["signal_type", "amount"], ascending=[True, False])


def _read_awards_cache(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "award_date" in df.columns:
        df["award_date"] = pd.to_datetime(df["award_date"], errors="coerce")
    return df


def _latest_stale_cache(max_age_days: int = CACHE_MAX_AGE_DAYS):
    """Most recent awards_*.csv within max_age_days (excluding today's, which
    would already have been used by the fresh-cache check). Returns
    (path, age_days) or None."""
    if not OUTPUT_DIR.exists():
        return None
    now = datetime.utcnow()
    best = None
    for p in OUTPUT_DIR.glob("awards_*.csv"):
        try:
            day = datetime.strptime(p.stem.replace("awards_", ""), "%Y%m%d")
        except ValueError:
            continue
        age = (now - day).days
        if 0 < age <= max_age_days and (best is None or day > best[1]):
            best = (p, day, age)
    return (best[0], best[2]) if best else None


def run(days: int = 14, min_amount: float = 1_000_000):
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y%m%d")
    cache_path = OUTPUT_DIR / f"awards_{today}.csv"

    if cache_path.exists():
        print(f"[CACHE] fresh — using today's cached contracts -> {cache_path}")
        return _read_awards_cache(cache_path)

    print(f"Scanning gov contracts | {start_date} -> {end_date} | Min: ${min_amount:,.0f}\n")

    all_ticker_rows = []
    errors = []
    attempted = 0

    for ticker, company_names in TICKER_TO_COMPANY.items():
        if not company_names:
            continue

        attempted += len(company_names)
        print(f"  {ticker}: searching {company_names}...")
        df = fetch_contracts(company_names, start_date, end_date, min_amount, errors=errors)

        if df.empty:
            print(f"    No contracts >= ${min_amount:,.0f}")
            continue

        df["ticker"] = ticker
        total = df["amount"].sum()
        print(f"    Found {len(df)} contracts | Total: ${total:,.0f}")
        all_ticker_rows.append(df)

    if attempted > 0 and len(errors) == attempted:
        # every single fetch failed (after retries) -> fall back to a recent cache
        stale = _latest_stale_cache()
        if stale is not None:
            path, age = stale
            print(f"[CACHE] stale-cache — all fetches failed, using {path.name} ({age}d old)")
            return _read_awards_cache(path)
        print("[CACHE] no-data — all fetches failed and no cache within "
              f"{CACHE_MAX_AGE_DAYS} days")
        return pd.DataFrame()

    if not all_ticker_rows:
        print("\nNo contracts found for any watchlist ticker.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["ticker", "signal_type"]).to_csv(cache_path, index=False)
        print(f"[CACHE] fresh — saved (no contracts today) -> {cache_path}")
        return pd.DataFrame()

    result = pd.concat(all_ticker_rows, ignore_index=True)
    result = result.sort_values(["amount"], ascending=False)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"GOV CONTRACTS — WATCHLIST HITS (last {days} days, min ${min_amount:,.0f})")
    print("=" * 70)

    display_cols = [c for c in ["ticker", "signal_type", "company", "amount", "award_date", "agency", "description"] if c in result.columns]
    display = result[display_cols].copy()
    display["amount"] = display["amount"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    display["award_date"] = display["award_date"].dt.strftime("%Y-%m-%d")
    if "description" in display.columns:
        display["description"] = display["description"].str[:50]
    print(display.to_string(index=False))

    # Signal summary: tickers with large contract activity
    print("\n--- Signal Summary ---")
    summary = result.groupby("ticker").agg(
        contracts=("award_id", "count"),
        total_value=("amount", "sum"),
        largest=("amount", "max"),
    ).sort_values("total_value", ascending=False)
    summary["total_value"] = summary["total_value"].apply(lambda x: f"${x:,.0f}")
    summary["largest"] = summary["largest"].apply(lambda x: f"${x:,.0f}")
    print(summary.to_string())

    # Save (also serves as today's cache — see fresh-cache check above)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    print(f"[CACHE] fresh — saved -> {cache_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gov contract watcher for watchlist tickers")
    parser.add_argument("--days", default=14, type=int, help="Lookback days (contracts reported within 3-7 days of award)")
    parser.add_argument("--min_amount", default=1_000_000, type=float, help="Minimum contract value to include")
    args = parser.parse_args()
    run(days=args.days, min_amount=args.min_amount)
