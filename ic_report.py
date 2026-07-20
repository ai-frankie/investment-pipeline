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

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

FACTOR_LOG = Path("output/factor_log.csv")
FACTORS = ["forecast_edge", "path_consistency", "vol_context",
           "trend_alignment", "lt_quality", "contract_signal_score",
           "news_sent", "kronos_fwd_ret", "score", "adj_score",
           "congress_mod", "insider_mod"]
DECAY_IC = 0.02        # rolling IC below this...
DECAY_WEEKS = 8        # ...for this many consecutive weeks = decay alarm


def spearman_ic(pred: pd.Series, realized: pd.Series) -> float:
    m = pd.concat([pred, realized], axis=1).dropna()
    if len(m) < 10:
        return np.nan
    return float(m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank()))


def spearman_ic_pvalue(pred: pd.Series, realized: pd.Series) -> tuple[float, float, int]:
    """IC + two-sided p-value via scipy (required for FDR correction downstream —
    a rank-corr-only IC has no p-value to correct). Returns (ic, pvalue, n)."""
    m = pd.concat([pred, realized], axis=1).dropna()
    n = len(m)
    if n < 10:
        return np.nan, np.nan, n
    res = spearmanr(m.iloc[:, 0], m.iloc[:, 1])
    return float(res.statistic), float(res.pvalue), n


def _entry_index(close_index, run_date) -> int:
    """Index of the earliest realistic fill session (next trading day after
    run_date). If run_date itself is a trading day, entry is the NEXT bar;
    if run_date is a weekend/holiday, searchsorted already lands on the next
    session — use it as-is. May return len(close_index): caller must bounds-check."""
    ts = pd.Timestamp(run_date)
    idx = close_index.searchsorted(ts)
    if idx < len(close_index) and close_index[idx] == ts.normalize():
        return idx + 1
    return idx


def load_with_forward_returns(horizon_days: int) -> pd.DataFrame:
    if not FACTOR_LOG.exists():
        raise SystemExit(f"No decision log yet at {FACTOR_LOG} — run the pipeline first.")
    log = pd.read_csv(FACTOR_LOG, parse_dates=["run_date"])
    log = log.dropna(subset=["close_at_score"])
    log = log.drop_duplicates(subset=["run_date", "ticker"], keep="last").reset_index(drop=True)

    tickers = sorted(log["ticker"].unique())
    start = (log["run_date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, interval="1d",
                     progress=False, auto_adjust=True, group_by="ticker")

    fwd = []
    fetch_failed = set()
    for _, row in log.iterrows():
        t = row["ticker"]
        try:
            tpx = px[t] if isinstance(px.columns, pd.MultiIndex) else px
            close = tpx["Close"].dropna()
            open_prices = tpx["Open"].reindex(close.index)
            entry_idx = _entry_index(close.index, row["run_date"])
            if entry_idx >= len(close) or entry_idx + horizon_days >= len(close):
                fwd.append(np.nan)  # no next session yet / future not realized
                continue
            entry_px = float(open_prices.iloc[entry_idx])   # fill at next-session OPEN, like ledger.py
            if not np.isfinite(entry_px) or entry_px <= 0:
                entry_px = float(close.iloc[entry_idx])     # fallback: next-session close
            fwd.append(float(close.iloc[entry_idx + horizon_days]) / entry_px - 1.0)
        except Exception:
            fwd.append(np.nan)
            fetch_failed.add(t)
    log[f"fwd_{horizon_days}d"] = fwd
    if fetch_failed:
        print(f"WARNING: {len(fetch_failed)} ticker(s) excluded from IC due to data-fetch failure "
              f"(possible delisting): {sorted(fetch_failed)}")
    return log


def regime_ic_table(log: pd.DataFrame, realized: pd.Series) -> None:
    """Purely diagnostic per-regime IC split (Phase E task E5). Factors can
    behave differently calm vs stressed; this just surfaces that, it does
    not feed any weight change — see learn_weights.py --regime and
    pipeline.py's reserved (not activated) factor_weights:"regime" for the
    data-gated activation path."""
    if "vix" not in log.columns:
        print("\n(per-regime IC skipped: no vix column)")
        return
    buckets = {"calm (vix<20)": log["vix"] < 20, "stressed (vix>=20)": log["vix"] >= 20}
    print(f"\n{'-'*60}\nPer-regime IC (diagnostic only)\n{'-'*60}")
    for name, mask in buckets.items():
        sub, sub_realized = log[mask], realized[mask]
        n_days = sub["run_date"].nunique()
        print(f"\n{name}: {n_days} run_dates, {len(sub)} rows")
        if n_days == 0:
            continue
        rows = []
        for f in FACTORS:
            if f not in sub.columns:
                continue
            ic = spearman_ic(sub[f], sub_realized)
            rows.append({"factor": f, "IC": round(ic, 3) if np.isfinite(ic) else ic,
                        "n": int(sub[f].notna().sum())})
        if rows:
            print(pd.DataFrame(rows).sort_values("IC", ascending=False).to_string(index=False))


def report(horizon_days: int):
    log = load_with_forward_returns(horizon_days)
    realized = log[f"fwd_{horizon_days}d"]
    n = int(realized.notna().sum())
    print(f"\n{'='*60}\nIC REPORT — {horizon_days}d forward horizon  ({n} realized obs)\n{'='*60}")
    if n < 10:
        print("Fewer than 10 realized observations — keep logging, come back later.")
        return

    if not _HAVE_SCIPY:
        print("scipy not installed — cannot compute p-values / FDR-corrected significance "
              "(pip install scipy). Falling back to uncorrected IC only.")
        rows = []
        for f in FACTORS:
            if f not in log.columns:
                continue
            rows.append({"factor": f,
                         "IC": round(spearman_ic(log[f], realized), 3),
                         "n": int(log[f].notna().sum())})
        out = pd.DataFrame(rows).sort_values("IC", ascending=False)
        print(out.to_string(index=False))
        print("\nGuide: |IC| > 0.05 = pulling weight, ~0 = dead weight, negative = inverted. "
              "(uncorrected — scipy unavailable)")
    else:
        from statsmodels.stats.multitest import multipletests

        rows = []
        for f in FACTORS:
            if f not in log.columns:
                continue
            ic, p, nf = spearman_ic_pvalue(log[f], realized)
            rows.append({"factor": f, "IC": round(ic, 3) if np.isfinite(ic) else ic,
                         "p": p, "n": nf})
        out = pd.DataFrame(rows)
        out["significant"] = False
        valid = out["p"].notna()
        if valid.sum() > 0:
            rejected, _, _, _ = multipletests(out.loc[valid, "p"], alpha=0.05, method="fdr_bh")
            out.loc[valid, "significant"] = rejected
        out = out.sort_values("IC", ascending=False)
        print(out.to_string(index=False))
        print("\nGuide: |IC| > 0.05 is UNCORRECTED — see 'significant' (FDR/Benjamini-Hochberg "
              "@ alpha=0.05 across all factors tested this run) for the multiple-testing-corrected read.")

    # Kronos decay check: weekly rolling IC of forecast edge vs realized.
    # Only meaningful at horizon_days == 5 — weekly sampling of a 5d horizon is
    # near non-overlapping; at 21d the windows overlap ~4x and the rolling IC
    # is dominated by autocorrelation, not genuine week-to-week decay.
    if horizon_days != 5:
        print(f"\nKronos decay check skipped at {horizon_days}d (overlapping windows).")
    else:
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

    regime_ic_table(log, realized)

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
