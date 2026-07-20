"""
build_meta_dataset.py
Phase E task E7: meta-label dataset extractor.

Meta-labeling: a second model that predicts WHEN the primary signal pays,
from context (VIX, regime, dispersion, news, earnings distance). At ~300
rows today, training now = overfit garbage. This script only builds the
dataset + prints a readiness verdict; training is a future task unlocked
once the verdict is READY (n >= 1000).

Reuses ic_report.load_with_forward_returns() (post-audit: next-session-open
fill anchor) for honest labels, so labels here are exactly as trustworthy
as the numbers in ic_report.py.

Usage:
    python build_meta_dataset.py
    python build_meta_dataset.py --horizon 5
"""

import argparse
from pathlib import Path

import pandas as pd

from ic_report import load_with_forward_returns

HORIZON = 21          # forward horizon, trading days (matches learn_weights.py's label)
COST = 0.0010          # 10 bps round-trip cost hurdle
READY_N = 1000
OUT = Path("output/meta_dataset.csv")

# Features already logged per row after E1 ("path_dispersion"/"n_paths"/
# "news_sent") and E4 ("congress_mod"/"insider_mod"); "adj_score" is the
# E3 cross-sectional rank.
FEATURES = ["vix", "regime_ok", "path_dispersion", "n_paths", "news_sent",
            "days_to_earnings", "macro_event", "forecast_edge", "vol_context",
            "adj_score", "congress_mod", "insider_mod"]


def label_rows(log: pd.DataFrame, fwd: pd.Series) -> pd.DataFrame:
    """1 if an actionable row's honest forward return clears costs, else 0.
    Actionable = action == "BUY" OR (pre-gate) score >= 0.7 — a row the
    scorer rated a real signal even if a downstream gate (regime/earnings/
    rank/news) downgraded the displayed action to HOLD."""
    out = log.copy()
    actionable = out["action"] == "BUY"
    if "score" in out.columns:
        actionable = actionable | (out["score"] >= 0.7)
    profitable = fwd > COST
    out["label"] = (actionable & profitable).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description="Meta-label dataset extractor")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="Forward horizon in trading days (default: 21)")
    args = parser.parse_args()

    log = load_with_forward_returns(args.horizon)
    fwd = log[f"fwd_{args.horizon}d"]
    labeled = label_rows(log, fwd)

    cols = [c for c in FEATURES if c in labeled.columns]
    missing = [c for c in FEATURES if c not in labeled.columns]
    if missing:
        print(f"[META] feature(s) not in factor_log.csv yet (pre-E1/E4 rows lack them): {missing}")

    keep = ["run_date", "ticker"] + cols + ["label"]
    dataset = labeled.dropna(subset=cols + [f"fwd_{args.horizon}d"])[keep]

    OUT.parent.mkdir(exist_ok=True)
    dataset.to_csv(OUT, index=False)

    n = len(dataset)
    verdict = (f"READY (n={n}>={READY_N})" if n >= READY_N
               else f"NOT READY (n={n}, need ~{READY_N - n} more labeled rows)")
    print(f"[META] {n} labeled rows -> {OUT}")
    print(f"[META] verdict: {verdict}")


if __name__ == "__main__":
    main()
