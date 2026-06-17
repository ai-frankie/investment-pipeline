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
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

OUTPUT_DIR = Path("output/contracts")

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


def fetch_contracts(company_names: list, start_date: str, end_date: str, min_amount: float) -> pd.DataFrame:
    """Fetch contract awards for a list of company name variants."""
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
            resp = requests.post(BASE_URL, json=payload, timeout=15)
            resp.raise_for_status()
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


def run(days: int = 14, min_amount: float = 1_000_000):
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"Scanning gov contracts | {start_date} -> {end_date} | Min: ${min_amount:,.0f}\n")

    all_ticker_rows = []

    for ticker, company_names in TICKER_TO_COMPANY.items():
        if not company_names:
            continue

        print(f"  {ticker}: searching {company_names}...")
        df = fetch_contracts(company_names, start_date, end_date, min_amount)

        if df.empty:
            print(f"    No contracts >= ${min_amount:,.0f}")
            continue

        df["ticker"] = ticker
        total = df["amount"].sum()
        print(f"    Found {len(df)} contracts | Total: ${total:,.0f}")
        all_ticker_rows.append(df)

    if not all_ticker_rows:
        print("\nNo contracts found for any watchlist ticker.")
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

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"contracts_{today}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gov contract watcher for watchlist tickers")
    parser.add_argument("--days", default=14, type=int, help="Lookback days (contracts reported within 3-7 days of award)")
    parser.add_argument("--min_amount", default=1_000_000, type=float, help="Minimum contract value to include")
    args = parser.parse_args()
    run(days=args.days, min_amount=args.min_amount)
