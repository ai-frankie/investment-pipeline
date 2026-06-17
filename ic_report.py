"""
ic_report.py
Model-decay dashboard: joins realized forward returns onto the decision log
(output/factor_log.csv, written by every pipeline run) and reports per-factor
information coefficients.

Answers two questions:
  1. Which factors actually predict forward returns? (feeds future weight learning)
  2. Is Kronos decaying? Rolling IC below threshold for several consecutive
     windows = stop trusting the model for new entries.

Needs history to be useful — ICs stabilize after ~8+ weeks of daily logs.

Usage:
    python ic_report.py                 # 5d and 21d horizons
    python ic_report.py --horizon 21
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FACTOR_LOG = Path("output/factor_log.csv")
FACTORS = ["forecast_edge", "path_consistency", "vol_context",
           "trend_alignment", "lt_quality", "contract_signal_score",
           "news_sent", "kronos_fwd_ret", "score", "adj_score"]
DECAY_IC = 0.02        # rolling IC below this...
DECAY_WEEKS = 8        # ...for this many consecutive weeks = decay alarm


def spearman_ic(pred: pd.Series, realized: pd.Series) -> float:
    m = pd.concat([pred, realized], axis=1).dropna()
    if len(m) < 10:
        return np.nan
    return float(m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank()))


def load_with_forward_returns(horizon_days: int) -> pd.DataFrame:
    if not FACTOR_LOG.exists():
        raise SystemExit(f"No decision log yet at {FACTOR_LOG} — run the pipeline first.")
    log = pd.read_csv(FACTOR_LOG, parse_dates=["run_date"])
    log = log.dropna(subset=["close_at_score"])

    tickers = sorted(log["ticker"].unique())
    start = (log["run_date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, interval="1d",
                     progress=False, auto_adjust=True, group_by="ticker")

    fwd = []
    for _, row in log.iterrows():
        t = row["ticker"]
        try:
            close = px[t]["Close"].dropna() if isinstance(px.columns, pd.MultiIndex) else px["Close"].dropna()
            idx = close.index.searchsorted(row["run_date"])
            if idx + horizon_days >= len(close):
                fwd.append(np.nan)  # future not realized yet
                continue
            fwd.append(float(close.iloc[idx + horizon_days]) / float(close.iloc[idx]) - 1.0)
        except Exception:
            fwd.append(np.nan)
    log[f"fwd_{horizon_days}d"] = fwd
    return log


def report(horizon_days: int):
    log = load_with_forward_returns(horizon_days)
    realized = log[f"fwd_{horizon_days}d"]
    n = int(realized.notna().sum())
    print(f"\n{'='*60}\nIC REPORT — {horizon_days}d forward horizon  ({n} realized obs)\n{'='*60}")
    if n < 10:
        print("Fewer than 10 realized observations — keep logging, come back later.")
        return

    rows = []
    for f in FACTORS:
        if f not in log.columns:
            continue
        rows.append({"factor": f,
                     "IC": round(spearman_ic(log[f], realized), 3),
                     "n": int(log[f].notna().sum())})
    out = pd.DataFrame(rows).sort_values("IC", ascending=False)
    print(out.to_string(index=False))
    print("\nGuide: |IC| > 0.05 = pulling weight, ~0 = dead weight, negative = inverted.")

    # Kronos decay check: weekly rolling IC of forecast edge vs realized
    k = log.dropna(subset=["kronos_fwd_ret", f"fwd_{horizon_days}d"]).copy()
    if len(k) >= 30:
        k["week"] = k["run_date"].dt.to_period("W")
        weekly_ic = k.groupby("week").apply(
            lambda g: spearman_ic(g["kronos_fwd_ret"], g[f"fwd_{horizon_days}d"]),
            include_groups=False).dropna()
        if len(weekly_ic) >= DECAY_WEEKS:
            rolling = weekly_ic.rolling(DECAY_WEEKS).mean()
            latest = float(rolling.iloc[-1])
            consec_low = int((weekly_ic.tail(DECAY_WEEKS) < DECAY_IC).sum())
            print(f"\nKronos rolling {DECAY_WEEKS}w IC: {latest:+.3f}")
            if consec_low >= DECAY_WEEKS:
                print(f"*** DECAY ALARM: IC < {DECAY_IC} for {DECAY_WEEKS} straight weeks "
                      f"— stop trusting Kronos for new entries (run with --no-kronos). ***")
        else:
            print(f"\nKronos decay check: needs {DECAY_WEEKS}+ weeks of logs "
                  f"(have {len(weekly_ic)}).")

    ts = pd.Timestamp.now().strftime("%Y%m%d")
    out_path = Path("output") / f"ic_report_{horizon_days}d_{ts}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor IC + model decay report")
    parser.add_argument("--horizon", type=int, choices=[5, 21], default=None,
                        help="Forward horizon in trading days (default: both)")
    args = parser.parse_args()
    for h in ([args.horizon] if args.horizon else [5, 21]):
        report(h)
