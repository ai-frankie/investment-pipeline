"""
health_check.py
Phase E task E8: codified kill/keep criteria, weekly health check.

Read-only report. Every verdict below is a printed RECOMMENDATION for
Frank -- this script never kills a factor, promotes a modifier, activates
a gate, or touches config.json. It only reads factor_log.csv, the
intraday proposal/ledger files, and (if present) output/meta_dataset.csv,
and prints evidence (IC, n, dates) behind every call.

Rules (see docs/superpowers/plans/2026-07-11-phase-e-alpha-upgrades.md,
Task E8):
  - Each factor: FDR-significant IC <= 0 in the last 2 consecutive
    calendar months of logging -> KILL-CANDIDATE (needs Frank)
  - Intraday: >=10 post-fix scan days AND (score-vs-next-scan IC < 0.02
    OR net pnl_dollars < 0 over >=20 closed trades) -> SHUT OFF SCHEDULE
  - Kronos daily: existing 5d decay alarm from ic_report.py -- passthrough
  - congress/insider (log mode): >=60 run_dates, FDR-significant,
    |IC| >= 0.03 -> PROMOTE-CANDIDATE
  - dispersion gate (log mode): below-threshold IC - above-threshold IC
    >= 0.03 -> ACTIVATE-CANDIDATE
  - Meta dataset: n >= 1000 -> TRAIN-READY
  - Regime buckets: each >= 60 run_dates -> FIT-READY

Usage:
    python health_check.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ic_report import (FACTOR_LOG, FACTORS as IC_FACTORS, DECAY_IC, DECAY_WEEKS,
                       load_with_forward_returns, spearman_ic, spearman_ic_pvalue)

KILL_HORIZON = 21          # matches learn_weights.py's label horizon
ALTDATA_MIN_RUN_DATES = 60
DISPERSION_MIN_RUN_DATES = 10
INTRADAY_MIN_DAYS = 10
INTRADAY_MIN_TRADES = 20
INTRADAY_IC_FLOOR = 0.02
META_READY_N = 1000
REGIME_MIN_RUN_DATES = 60


def _p(lines: list, msg: str) -> None:
    print(msg)
    lines.append(msg)


def factor_kill_check(lines: list, horizon_days: int = KILL_HORIZON) -> None:
    _p(lines, f"\n--- Factor kill-check ({horizon_days}d horizon, last 2 calendar months) ---")
    log = load_with_forward_returns(horizon_days)
    realized = log[f"fwd_{horizon_days}d"]
    log = log.assign(month=log["run_date"].dt.to_period("M"))
    months = sorted(log["month"].dropna().unique())
    if len(months) < 2:
        _p(lines, f"insufficient monthly history: {len(months)} month(s) logged, need 2 consecutive")
        return

    verdicts = {}
    for m in months[-2:]:
        mask = log["month"] == m
        sub, sub_realized = log[mask], realized[mask]
        rows = []
        for f in IC_FACTORS:
            if f not in sub.columns:
                continue
            ic, p, n = spearman_ic_pvalue(sub[f], sub_realized)
            rows.append({"factor": f, "IC": ic, "p": p, "n": n})
        fdf = pd.DataFrame(rows)
        fdf["significant"] = False
        valid = fdf["p"].notna()
        if valid.sum() > 0:
            rejected, _, _, _ = multipletests(fdf.loc[valid, "p"], alpha=0.05, method="fdr_bh")
            fdf.loc[valid, "significant"] = rejected
        verdicts[m] = fdf.set_index("factor")

    m1, m2 = months[-2:]
    for f in IC_FACTORS:
        if f not in verdicts[m1].index or f not in verdicts[m2].index:
            continue
        v1, v2 = verdicts[m1].loc[f], verdicts[m2].loc[f]
        kill = (bool(v1["significant"]) and np.isfinite(v1["IC"]) and v1["IC"] <= 0
                and bool(v2["significant"]) and np.isfinite(v2["IC"]) and v2["IC"] <= 0)
        verdict = "KILL-CANDIDATE (needs Frank)" if kill else "ok"
        _p(lines, f"  {f}: {m1} IC={v1['IC']:.3f} sig={bool(v1['significant'])} (n={int(v1['n'])})  "
                  f"{m2} IC={v2['IC']:.3f} sig={bool(v2['significant'])} (n={int(v2['n'])})  -> {verdict}")


def kronos_decay_passthrough(lines: list) -> None:
    _p(lines, "\n--- Kronos daily decay (5d horizon, passthrough from ic_report.py) ---")
    log = load_with_forward_returns(5)
    k = log.dropna(subset=["kronos_fwd_ret", "fwd_5d"]).copy()
    if len(k) < 30:
        _p(lines, f"insufficient data ({len(k)} realized obs, need 30+)")
        return
    k["week"] = k["run_date"].dt.to_period("W")
    weekly_ic = k.groupby("week").apply(
        lambda g: spearman_ic(g["kronos_fwd_ret"], g["fwd_5d"]), include_groups=False).dropna()
    if len(weekly_ic) < DECAY_WEEKS:
        _p(lines, f"needs {DECAY_WEEKS}+ weeks of logs (have {len(weekly_ic)})")
        return
    rolling = weekly_ic.rolling(DECAY_WEEKS).mean()
    latest = float(rolling.iloc[-1])
    consec_low = int((weekly_ic.tail(DECAY_WEEKS) < DECAY_IC).sum())
    _p(lines, f"rolling {DECAY_WEEKS}w IC: {latest:+.3f}")
    if consec_low >= DECAY_WEEKS:
        _p(lines, f"*** DECAY ALARM: IC < {DECAY_IC} for {DECAY_WEEKS} straight weeks "
                  f"-- stop trusting Kronos for new entries (run with --no-kronos). ***")
    else:
        _p(lines, "OK -- no decay alarm")


def intraday_health_check(lines: list) -> None:
    """score-vs-next-scan IC is a proxy (no honest forward-return pipe exists
    for intraday scans today, unlike the daily pipeline's next-open anchor):
    for each ticker, does its score at one scan predict its own price move
    to its NEXT logged scan. Documented here as an interpretation call, not
    a plan-specified formula."""
    _p(lines, "\n--- Intraday system ---")
    # Only the primary per-day scan files (proposals_intraday_YYYYMMDD.csv) --
    # skip the odd _HHMM debug-run variants seen in output/intraday/.
    day_pattern = re.compile(r"proposals_intraday_\d{8}\.csv$")
    day_files = sorted(f for f in Path("output/intraday").glob("proposals_intraday_*.csv")
                       if day_pattern.match(f.name))
    if not day_files:
        _p(lines, "no intraday proposal files yet")
        return

    frames = []
    for f in day_files:
        df = pd.read_csv(f)
        if "scan_time" in df.columns and "last_close" in df.columns and "score" in df.columns:
            df["scan_date"] = pd.to_datetime(df["scan_time"]).dt.date
            frames.append(df)
    if not frames:
        _p(lines, "no usable intraday scan data (missing scan_time/last_close/score)")
        return

    all_df = pd.concat(frames, ignore_index=True)
    n_days = all_df["scan_date"].nunique()
    _p(lines, f"trading days logged: {n_days}")

    rank_ic = np.nan
    if n_days >= INTRADAY_MIN_DAYS:
        all_df = all_df.sort_values(["ticker", "scan_time"])
        all_df["next_close"] = all_df.groupby("ticker")["last_close"].shift(-1)
        m = all_df.dropna(subset=["score", "next_close", "last_close"])
        m = m[m["last_close"] > 0]
        fwd_ret = m["next_close"] / m["last_close"] - 1.0
        rank_ic = spearman_ic(m["score"], fwd_ret)
    ic_s = f"{rank_ic:.3f}" if np.isfinite(rank_ic) else "n/a"
    _p(lines, f"score-vs-next-scan IC (proxy): {ic_s}")

    ct_path = Path("ledger/intraday/closed_trades.csv")
    net_pnl, n_trades = None, 0
    if ct_path.exists():
        ct = pd.read_csv(ct_path)
        if "pnl_dollars" in ct.columns:
            pnl = ct["pnl_dollars"].dropna()
            n_trades = len(pnl)
            net_pnl = float(pnl.sum()) if n_trades else None
    pnl_s = f"{net_pnl:.2f}" if net_pnl is not None else "n/a"
    _p(lines, f"closed trades with realized PnL: {n_trades}, net pnl_dollars: {pnl_s}")

    if n_days >= INTRADAY_MIN_DAYS and n_trades >= INTRADAY_MIN_TRADES:
        shut_off = ((np.isfinite(rank_ic) and rank_ic < INTRADAY_IC_FLOOR)
                    or (net_pnl is not None and net_pnl < 0))
        _p(lines, "SHUT OFF SCHEDULE (needs Frank)" if shut_off else "ok -- within bounds")
    else:
        _p(lines, f"insufficient data for verdict (need >={INTRADAY_MIN_DAYS} trading days AND "
                  f">={INTRADAY_MIN_TRADES} closed trades; have {n_days} days, {n_trades} trades)")


def altdata_promotion_check(lines: list, horizon_days: int = KILL_HORIZON) -> None:
    _p(lines, f"\n--- Alt-data promotion (congress/insider, log mode, {horizon_days}d) ---")
    log = load_with_forward_returns(horizon_days)
    realized = log[f"fwd_{horizon_days}d"]

    results = {}
    for f in ("congress_mod", "insider_mod"):
        if f not in log.columns:
            _p(lines, f"  {f}: column missing")
            continue
        n_days = log.loc[log[f].notna(), "run_date"].nunique()
        ic, p, n = spearman_ic_pvalue(log[f], realized)
        results[f] = (ic, p, n, n_days)
    if not results:
        return

    idx = [k for k, v in results.items() if np.isfinite(v[1])]
    sig = {k: False for k in results}
    if idx:
        rejected, _, _, _ = multipletests([results[k][1] for k in idx], alpha=0.05, method="fdr_bh")
        for k, r in zip(idx, rejected):
            sig[k] = bool(r)

    for f, (ic, p, n, n_days) in results.items():
        promote = (n_days >= ALTDATA_MIN_RUN_DATES and sig[f]
                   and np.isfinite(ic) and abs(ic) >= 0.03)
        verdict = "PROMOTE-CANDIDATE" if promote else "not yet"
        ic_s = f"{ic:.3f}" if np.isfinite(ic) else "n/a"
        _p(lines, f"  {f}: IC={ic_s} n={n} run_dates={n_days} significant={sig[f]} -> {verdict}")


def dispersion_activation_check(lines: list, horizon_days: int = KILL_HORIZON) -> None:
    _p(lines, f"\n--- Dispersion gate activation ({horizon_days}d) ---")
    with open("config.json") as fh:
        cfg = json.load(fh)
    max_rel = cfg.get("dispersion_max_rel", 3.0)

    log = load_with_forward_returns(horizon_days)
    realized = log[f"fwd_{horizon_days}d"]
    d = log.dropna(subset=["path_dispersion", "kronos_fwd_ret"])
    n_days = d["run_date"].nunique()
    if n_days < DISPERSION_MIN_RUN_DATES:
        _p(lines, f"insufficient dispersion history: {n_days} run_dates (<{DISPERSION_MIN_RUN_DATES})")
        return

    rel = d["path_dispersion"] / d["kronos_fwd_ret"].abs().clip(lower=1e-4)
    below, above = d[rel <= max_rel], d[rel > max_rel]
    below_ic = spearman_ic(below["kronos_fwd_ret"], realized.loc[below.index])
    above_ic = spearman_ic(above["kronos_fwd_ret"], realized.loc[above.index])
    below_s = f"{below_ic:.3f}" if np.isfinite(below_ic) else "n/a"
    above_s = f"{above_ic:.3f}" if np.isfinite(above_ic) else "n/a"
    _p(lines, f"below-threshold (<= {max_rel}) IC={below_s} (n={len(below)}), "
              f"above-threshold IC={above_s} (n={len(above)})")
    if np.isfinite(below_ic) and np.isfinite(above_ic) and (below_ic - above_ic) >= 0.03:
        _p(lines, "ACTIVATE-CANDIDATE")
    else:
        _p(lines, "not yet")


def meta_dataset_check(lines: list) -> None:
    _p(lines, "\n--- Meta dataset ---")
    path = Path("output/meta_dataset.csv")
    if not path.exists():
        _p(lines, "not built yet -- run build_meta_dataset.py")
        return
    n = len(pd.read_csv(path))
    verdict = "TRAIN-READY" if n >= META_READY_N else "not ready"
    _p(lines, f"n={n} -> {verdict}")


def regime_bucket_check(lines: list) -> None:
    _p(lines, "\n--- Regime bucket fit-readiness ---")
    if not FACTOR_LOG.exists():
        _p(lines, "no factor_log.csv yet")
        return
    log = pd.read_csv(FACTOR_LOG, parse_dates=["run_date"])
    if "vix" not in log.columns:
        _p(lines, "no vix column")
        return
    for name, mask in {"calm (vix<20)": log["vix"] < 20, "stressed (vix>=20)": log["vix"] >= 20}.items():
        n_days = log.loc[mask, "run_date"].nunique()
        verdict = "FIT-READY" if n_days >= REGIME_MIN_RUN_DATES else "not ready"
        _p(lines, f"  {name}: {n_days} run_dates -> {verdict}")


def main():
    lines = []
    _p(lines, f"{'='*65}")
    _p(lines, f"HEALTH CHECK — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    _p(lines, "Read-only. Every verdict is a recommendation for Frank -- this "
              "script never kills, promotes, gates, or activates anything itself.")
    _p(lines, f"{'='*65}")

    factor_kill_check(lines)
    kronos_decay_passthrough(lines)
    intraday_health_check(lines)
    altdata_promotion_check(lines)
    dispersion_activation_check(lines)
    meta_dataset_check(lines)
    regime_bucket_check(lines)

    out_dir = Path("output/health")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"health_{pd.Timestamp.now().strftime('%Y%m%d')}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
