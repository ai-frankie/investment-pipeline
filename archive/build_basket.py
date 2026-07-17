"""
build_basket.py
Builds the substitution pool: ~40 liquid names around the core watchlist,
diversified by sector caps and correlation clustering. PROPOSES ONLY — prints
the basket and writes output/basket_proposal.json; you decide what enters
config.json.

Rule (per spec):
  - 30-day average dollar volume >= $10M
  - hierarchical clustering (average linkage) on 1y daily return correlation,
    cut at correlation distance 0.35, max 2 names per cluster
  - max 6 names per sector
  - deliberate diversifiers vs tech+defense: utilities, staples, healthcare,
    plus a short-duration Treasury ballast sleeve

Usage:
    python build_basket.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Candidate seed pool with sectors (edit freely). Core watchlist included.
CANDIDATES = {
    # core — tech
    "NVDA": "tech", "AAPL": "tech", "MSFT": "tech", "AMZN": "tech",
    "META": "tech", "GOOGL": "tech", "AVGO": "tech", "CRM": "tech",
    # core — defense / gov services
    "PLTR": "defense", "CACI": "defense", "SAIC": "defense", "BAH": "defense",
    "LDOS": "defense", "LMT": "defense", "NOC": "defense", "GD": "defense",
    "RTX": "defense",
    # water / utilities
    "PHO": "utilities", "AWK": "utilities", "XYL": "utilities",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities", "XLU": "utilities",
    # consumer staples
    "PG": "staples", "KO": "staples", "PEP": "staples", "COST": "staples",
    "WMT": "staples", "XLP": "staples",
    # healthcare
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "XLV": "healthcare",
    # financials / industrials / energy breadth
    "JPM": "financials", "V": "financials", "BRK-B": "financials",
    "CAT": "industrials", "DE": "industrials", "HON": "industrials",
    "XOM": "energy", "CVX": "energy",
    # ballast
    "VGSH": "bonds", "SHY": "bonds", "AGG": "bonds",
}

ADV_FLOOR = 10_000_000     # 30d average dollar volume
CORR_DISTANCE_CUT = 0.35   # cluster cut: distance = 1 - correlation
MAX_PER_CLUSTER = 2
MAX_PER_SECTOR = 6
BASKET_SIZE = 40


def main():
    tickers = list(CANDIDATES)
    print(f"Screening {len(tickers)} candidates (1y daily, one batch download)...")
    data = yf.download(tickers, period="1y", interval="1d",
                       progress=False, auto_adjust=True, group_by="ticker")

    closes, advs = {}, {}
    for t in tickers:
        try:
            sub = data[t]
            close, vol = sub["Close"].dropna(), sub["Volume"].dropna()
            if len(close) < 200:
                continue
            adv = float((close * vol).tail(30).mean())
            if adv < ADV_FLOOR:
                print(f"  {t}: ADV ${adv/1e6:.1f}M < $10M floor — dropped")
                continue
            closes[t] = close
            advs[t] = adv
        except Exception as e:
            print(f"  {t}: {e}")

    rets = pd.DataFrame({t: c.pct_change() for t, c in closes.items()}).dropna()
    corr = rets.corr()
    dist = squareform((1 - corr).to_numpy(), checks=False)
    clusters = fcluster(linkage(dist, method="average"), t=CORR_DISTANCE_CUT,
                        criterion="distance")
    cluster_of = dict(zip(corr.columns, clusters))

    # rank by liquidity within constraints (score-based ranking can replace
    # this later — liquidity is the neutral default before factors exist
    # for non-watchlist names)
    ranked = sorted(closes, key=lambda t: -advs[t])
    basket, cluster_count, sector_count = [], {}, {}
    for t in ranked:
        cl, sec = cluster_of[t], CANDIDATES[t]
        if cluster_count.get(cl, 0) >= MAX_PER_CLUSTER:
            continue
        if sector_count.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        basket.append(t)
        cluster_count[cl] = cluster_count.get(cl, 0) + 1
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(basket) >= BASKET_SIZE:
            break

    df = pd.DataFrame([{"ticker": t, "sector": CANDIDATES[t],
                        "cluster": cluster_of[t], "adv_$M": round(advs[t] / 1e6, 0)}
                       for t in basket]).sort_values(["sector", "ticker"])
    print(f"\nProposed basket ({len(basket)} names, "
          f"{len(set(cluster_count))} correlation clusters):\n")
    print(df.to_string(index=False))
    print(f"\nSector counts: {sector_count}")

    out = Path("output/basket_proposal.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
                   "basket": basket,
                   "sectors": {t: CANDIDATES[t] for t in basket}}, f, indent=2)
    print(f"\nSaved -> {out}")
    print("Review, then add chosen names to config.json tickers — not automatic.")


if __name__ == "__main__":
    main()
