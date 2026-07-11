"""
learn_weights.py
Learns factor weights from the decision log via pooled panel ridge regression.
Replaces the equal-weight assumption once enough live data accumulates.

Leak-proof per spec:
  - Pooled across tickers (per-ticker models too noisy at this sample size)
  - Factors standardized CROSS-SECTIONALLY per date (relative ranking)
  - Walk-forward 3-fold by time, PURGED: training rows whose 21-day label
    window overlaps the test fold are dropped, +5-day embargo after each fold
  - Ridge alpha grid [0.1 .. 100], selected by mean out-of-sample Spearman IC
  - Refuses to run with <100 realized observations (come back in ~2 months)

Output: output/learned_weights.json — set "factor_weights": "learned" in
config.json to activate. Re-fit quarterly, not weekly (250 obs can't support
frequent re-tuning); or when ic_report.py shows sustained decay.

Usage:
    python learn_weights.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ic_report import load_with_forward_returns, spearman_ic

FACTORS = ["forecast_edge", "path_consistency", "vol_context",
           "trend_alignment", "lt_quality", "contract_signal_score"]
HORIZON = 21          # label: forward 21 trading days
EMBARGO = 5           # extra days after each test fold
ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100]
MIN_OBS = 100
OUT = Path("output/learned_weights.json")


def standardize_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date z-score of each factor across tickers (needs 3+ names/date)."""
    def z(g):
        if len(g) < 3:
            return g * np.nan
        sd = g.std(ddof=0)
        if sd == 0:
            return g * np.nan
        return (g - g.mean()) / sd
    out = df.copy()
    out[FACTORS] = df.groupby("run_date")[FACTORS].transform(z)
    return out.dropna(subset=FACTORS)


def purged_folds(dates: pd.Series, n_folds: int = 3):
    """Yield (train_idx, test_idx) with label-overlap purge + embargo.

    Purge/embargo bounds are exact positional trading-day lookups on the
    unique logged dates (not a calendar-day heuristic) — HORIZON and EMBARGO
    are trading-day counts, and the log itself is the only trading-day
    calendar we have (weekends/holidays are simply absent from it).
    """
    unique_days = np.sort(dates.unique())
    folds = np.array_split(unique_days, n_folds)
    for fold_days in folds:
        t0, t1 = fold_days.min(), fold_days.max()
        test = dates.isin(fold_days)
        pos0 = np.searchsorted(unique_days, t0)
        purge_start = unique_days[max(0, pos0 - HORIZON)]
        pos1 = np.searchsorted(unique_days, t1, side="right") - 1
        purge_end = unique_days[min(len(unique_days) - 1, pos1 + EMBARGO)]
        train = ~test & ~dates.between(purge_start, purge_end)
        yield train.to_numpy(), test.to_numpy()


def weights_from_coefs(coefs: np.ndarray, factors: list[str]) -> dict:
    """Positive-coefficient factors share weight; negative-IC factors are
    DROPPED (weight 0.0) so pipeline.py's weights.get(k, 0) > 0 filter
    excludes them, instead of silently inverting their signal."""
    pos = np.where(coefs > 0, coefs, 0.0)
    if pos.sum() == 0:
        raise SystemExit("All factors have non-positive coefficients — keeping equal weights.")
    return {f: round(float(c / pos.sum()), 4) for f, c in zip(factors, pos)}


def main():
    log = load_with_forward_returns(HORIZON)
    label = f"fwd_{HORIZON}d"
    df = log.dropna(subset=FACTORS + [label]).copy()
    n = len(df)
    print(f"Realized observations: {n}")
    if n < MIN_OBS:
        raise SystemExit(f"Need {MIN_OBS}+ realized obs to fit weights (have {n}). "
                         "Keep the pipeline logging daily — revisit in ~2 months.")

    df = standardize_cross_sectional(df)
    X, y, dates = df[FACTORS].to_numpy(), df[label].to_numpy(), df["run_date"]

    folds = list(purged_folds(dates))

    results = []
    for alpha in ALPHAS:
        fold_ics = []
        for train, test in folds:
            if train.sum() < 30 or test.sum() < 10:
                continue
            model = Ridge(alpha=alpha).fit(X[train], y[train])
            pred = model.predict(X[test])
            fold_ics.append(spearman_ic(pd.Series(pred), pd.Series(y[test])))
        if fold_ics:
            results.append({"alpha": alpha, "oos_ic_mean": float(np.nanmean(fold_ics)),
                            "oos_ic_per_fold": [round(x, 3) for x in fold_ics]})

    rdf = pd.DataFrame(results)
    print("\nAlpha grid (out-of-sample Spearman IC, purged folds — validates alpha selection only):")
    print(rdf[["alpha", "oos_ic_mean"]].to_string(index=False))

    best = rdf.loc[rdf["oos_ic_mean"].idxmax()]
    # multiple-testing sanity: tried len(ALPHAS) variants — demand the winner
    # clear a higher bar than "best of 7 looks positive"
    if best["oos_ic_mean"] < 0.03:
        print(f"\nBest alpha-selection OOS IC {best['oos_ic_mean']:.3f} < 0.03 after testing "
              f"{len(ALPHAS)} variants — too weak to trust. Keeping equal weights.")
        return

    # Honest holdout: the alpha grid above reused all 3 folds for CV, so its
    # IC only validates alpha SELECTION, not the final weights. Refit on data
    # strictly before the last fold's purge window and score only on that
    # never-trained fold — the only number that estimates how these exact
    # weights would generalize.
    train_last, test_last = folds[-1]
    if train_last.sum() < 30 or test_last.sum() < 10:
        print("\nFinal holdout fold too small to validate — keeping equal weights.")
        return
    final = Ridge(alpha=float(best["alpha"])).fit(X[train_last], y[train_last])
    holdout_pred = final.predict(X[test_last])
    holdout_ic = spearman_ic(pd.Series(holdout_pred), pd.Series(y[test_last]))
    print(f"\nFinal-weights holdout IC (never-trained fold): {holdout_ic:.3f}")
    if not np.isfinite(holdout_ic) or holdout_ic <= 0.03:
        print(f"Holdout IC <= 0.03 or undefined — weights not validated. Keeping equal weights.")
        return

    coefs = final.coef_
    weights = weights_from_coefs(coefs, FACTORS)
    signs = {f: int(np.sign(c)) for f, c in zip(FACTORS, coefs)}
    dropped = [f for f, s in signs.items() if s <= 0]

    out = {"fitted": pd.Timestamp.now().strftime("%Y-%m-%d"), "n_obs": n,
           "alpha": float(best["alpha"]),
           "alpha_selection_oos_ic": round(float(best["oos_ic_mean"]), 4),
           "final_weights_holdout_ic": round(float(holdout_ic), 4),
           "oos_ic_caveat": "validates alpha selection only",
           "variants_tested": len(ALPHAS), "weights": weights, "coef_signs": signs}
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nChosen alpha={best['alpha']}, final-weights holdout IC={holdout_ic:.3f}")
    print(f"Weights: {weights}")
    if dropped:
        print(f"Dropped from weights (weight=0, non-positive coefficient): {dropped}")
    print(f"\nSaved -> {OUT}")
    print('Activate with "factor_weights": "learned" in config.json. Refit quarterly.')


if __name__ == "__main__":
    main()
