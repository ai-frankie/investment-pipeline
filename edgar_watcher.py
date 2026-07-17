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
    """SEC fair-access pacing (0.3s before every attempt, unchanged) plus 3
    attempts / 1s/2s/4s backoff. Retries network errors and 5xx only — a
    4xx (e.g. bad CIK/accession) won't succeed on retry."""
    backoffs = (1, 2, 4)
    for attempt in range(len(backoffs)):
        time.sleep(0.3)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
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
    """All open-market insider buys for the ticker list, last N days.
    Cached per-ticker, per-day: a ticker whose submissions fetch fails is
    NOT marked settled — caching it as "no buys" would silently hide real
    signal for the rest of the day on a transient SEC error. The next call
    for that ticker retries it and merges the result into the day's cache.
    Success is tracked explicitly (sidecar), not inferred from row presence
    — a ticker with genuinely zero filings today is still a successful
    fetch and must not be re-hit every call."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    cache = OUTPUT_DIR / f"form4_{day}.csv"
    meta_path = OUTPUT_DIR / f"form4_{day}.meta.json"

    cols = ["ticker", "insider", "officer_or_dir", "date", "shares", "price", "dollars", "filed"]
    cached_df = pd.DataFrame(columns=cols)
    succeeded = set()
    if cache.exists():
        cached_df = pd.read_csv(cache, parse_dates=["date"])
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    succeeded = set(json.load(f).get("succeeded", []))
            except (json.JSONDecodeError, OSError):
                succeeded = set()

    wanted = {t.upper() for t in tickers}
    need_fetch = wanted - succeeded

    if not need_fetch:
        return cached_df[cached_df["ticker"].isin(wanted)]

    m = cik_map()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    new_rows = []
    newly_succeeded = set()
    for t in need_fetch:
        cik = m.get(t)
        if not cik:
            continue  # no CIK mapping -> nothing to fetch, not a failure to retry
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
                        row["ticker"] = t
                        row["filed"] = fdate
                        new_rows.append(row)
                except Exception as e:
                    print(f"[EDGAR] {t} {acc}: {e}")
            newly_succeeded.add(t)  # submissions list fetched OK -> settled for today
        except Exception as e:
            print(f"[EDGAR] {t}: {e}")  # not marked succeeded -> retried next call today

    new_df = pd.DataFrame(new_rows, columns=cols)
    if not new_df.empty:
        new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")

    if not cached_df.empty:
        combined = pd.concat([cached_df[~cached_df["ticker"].isin(newly_succeeded)], new_df],
                             ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(cache, index=False)
    with open(meta_path, "w") as f:
        json.dump({"succeeded": sorted(succeeded | newly_succeeded)}, f)

    return combined[combined["ticker"].isin(wanted)]


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
