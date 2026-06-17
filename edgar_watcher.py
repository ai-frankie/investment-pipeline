"""
edgar_watcher.py
SEC Form 4 insider-buy factor. Official EDGAR endpoints, no API key.

Signal spec (research-backed cluster-buy rule):
  Count ONLY open-market purchases: transactionCode == "P" and
  acquired/disposed == "A", excluding filings with the 10b5-1 plan checkbox
  (planned, not conviction). Codes A/M/F/S are compensation, exercises,
  tax withholding, sales — never buy signals.

  STRONG: >=3 distinct insiders buying within 30 days, >= $100k aggregate,
          at least one officer or director
  WEAK:   >=1 insider, >= $25k aggregate

Endpoints:
  ticker->CIK   https://www.sec.gov/files/company_tickers.json
  filings list  https://data.sec.gov/submissions/CIK##########.json
  Form 4 XML    https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}

SEC fair-access: declared User-Agent, ~0.3s between requests. Daily batch
use for ~15 tickers is far below the 10 req/s ceiling.

Usage:
    python edgar_watcher.py --tickers PLTR CACI NVDA --days 90
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "Frank Campos fcampos97@gmail.com"}
OUTPUT_DIR = Path("output/insiders")
CIK_CACHE = OUTPUT_DIR / "cik_map.json"

STRONG_BUYERS = 3
STRONG_DOLLARS = 100_000
WEAK_DOLLARS = 25_000
CLUSTER_DAYS = 30


def _get(url: str) -> requests.Response:
    time.sleep(0.3)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp


def cik_map() -> dict:
    """ticker -> zero-padded CIK, cached locally (changes rarely)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CIK_CACHE.exists():
        with open(CIK_CACHE) as f:
            return json.load(f)
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    m = {v["ticker"].upper(): f"{v['cik_str']:010d}" for v in data.values()}
    with open(CIK_CACHE, "w") as f:
        json.dump(m, f)
    return m


def _text(el, path: str):
    node = el.find(path)
    return node.text.strip() if node is not None and node.text else None


def parse_form4(xml_text: str) -> list[dict]:
    """Extract open-market buy rows from one Form 4 XML document."""
    root = ET.fromstring(xml_text)
    # 10b5-1 plan checkbox (post-2023 amendment) — planned trades, exclude
    if (_text(root, ".//aff10b5One") or "0") in ("1", "true"):
        return []

    owner = _text(root, ".//reportingOwner/reportingOwnerId/rptOwnerName") or "unknown"
    rel = root.find(".//reportingOwner/reportingOwnerRelationship")
    is_officer = rel is not None and (_text(rel, "isOfficer") in ("1", "true")
                                      or _text(rel, "isDirector") in ("1", "true"))

    rows = []
    for tx in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(tx, "transactionCoding/transactionCode")
        ad = _text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
        if code != "P" or ad != "A":
            continue
        shares = float(_text(tx, "transactionAmounts/transactionShares/value") or 0)
        price = float(_text(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        date = _text(tx, "transactionDate/value")
        rows.append({"insider": owner, "officer_or_dir": is_officer,
                     "date": date, "shares": shares, "price": price,
                     "dollars": round(shares * price, 0)})
    return rows


def fetch_insider_buys(tickers, days: int = 90) -> pd.DataFrame:
    """All open-market insider buys for the ticker list, last N days. Cached daily."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT_DIR / f"form4_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        return df[df["ticker"].isin([t.upper() for t in tickers])]

    m = cik_map()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    all_rows = []
    for t in tickers:
        cik = m.get(t.upper())
        if not cik:
            continue
        try:
            sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
            recent = sub["filings"]["recent"]
            for form, acc, doc, fdate in zip(recent["form"], recent["accessionNumber"],
                                             recent["primaryDocument"], recent["filingDate"]):
                if form != "4" or fdate < cutoff:
                    continue
                doc = doc.split("/")[-1]  # strip xslF345X05/ viewer prefix
                url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                       f"{acc.replace('-', '')}/{doc}")
                try:
                    for row in parse_form4(_get(url).text):
                        row["ticker"] = t.upper()
                        row["filed"] = fdate
                        all_rows.append(row)
                except Exception as e:
                    print(f"[EDGAR] {t} {acc}: {e}")
        except Exception as e:
            print(f"[EDGAR] {t}: {e}")

    df = pd.DataFrame(all_rows, columns=["ticker", "insider", "officer_or_dir",
                                         "date", "shares", "price", "dollars", "filed"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.to_csv(cache, index=False)
    return df


def get_insider_signals(tickers, days: int = CLUSTER_DAYS) -> dict:
    """{ticker: {"strength": "STRONG"|"WEAK", "buyers": n, "dollars": x}}"""
    df = fetch_insider_buys(tickers, days=max(days, 90))
    if df.empty:
        return {}
    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days)).tz_localize(None)
    df = df[df["date"] >= cutoff]
    out = {}
    for t, g in df.groupby("ticker"):
        buyers = g["insider"].nunique()
        dollars = float(g["dollars"].sum())
        has_officer = bool(g["officer_or_dir"].any())
        if buyers >= STRONG_BUYERS and dollars >= STRONG_DOLLARS and has_officer:
            out[t] = {"strength": "STRONG", "buyers": buyers, "dollars": dollars}
        elif dollars >= WEAK_DOLLARS:
            out[t] = {"strength": "WEAK", "buyers": buyers, "dollars": dollars}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC Form 4 insider-buy watcher")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    df = fetch_insider_buys(args.tickers, days=args.days)
    if df.empty:
        print("No open-market insider buys found.")
    else:
        print(df.sort_values("date", ascending=False).to_string(index=False))
        print("\nSignals:", get_insider_signals(args.tickers))
