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
    """Yield (train_idx, test_idx) with label-overlap purge + embargo."""
    unique_days = np.sort(dates.unique())
    folds = np.array_split(unique_days, n_folds)
    for fold_days in folds:
        t0, t1 = fold_days.min(), fold_days.max()
        test = dates.isin(fold_days)
        # purge: train rows whose 21d label window touches [t0, t1+embargo]
        purge_start = t0 - pd.Timedelta(days=int(HORIZON * 1.5))  # trading->calendar buffer
        purge_end = t1 + pd.Timedelta(days=EMBARGO)
        train = ~test & ~dates.between(purge_start, purge_end)
        yield train.to_numpy(), test.to_numpy()


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

    results = []
    for alpha in ALPHAS:
        fold_ics = []
        for train, test in purged_folds(dates):
            if train.sum() < 30 or test.sum() < 10:
                continue
            model = Ridge(alpha=alpha).fit(X[train], y[train])
            pred = model.predict(X[test])
            fold_ics.append(spearman_ic(pd.Series(pred), pd.Series(y[test])))
        if fold_ics:
            results.append({"alpha": alpha, "oos_ic_mean": float(np.nanmean(fold_ics)),
                            "oos_ic_per_fold": [round(x, 3) for x in fold_ics]})

    rdf = pd.DataFrame(results)
    print("\nAlpha grid (out-of-sample Spearman IC, purged folds):")
    print(rdf[["alpha", "oos_ic_mean"]].to_string(index=False))

    best = rdf.loc[rdf["oos_ic_mean"].idxmax()]
    # multiple-testing sanity: tried len(ALPHAS) variants — demand the winner
    # clear a higher bar than "best of 7 looks positive"
    if best["oos_ic_mean"] < 0.03:
        print(f"\nBest OOS IC {best['oos_ic_mean']:.3f} < 0.03 after testing "
              f"{len(ALPHAS)} variants — too weak to trust. Keeping equal weights.")
        return

    final = Ridge(alpha=float(best["alpha"])).fit(X, y)
    coefs = np.abs(final.coef_)
    if coefs.sum() == 0:
        raise SystemExit("Degenerate fit — all coefficients zero.")
    weights = {f: round(float(c / coefs.sum()), 4) for f, c in zip(FACTORS, coefs)}
    signs = {f: int(np.sign(c)) for f, c in zip(FACTORS, final.coef_)}

    out = {"fitted": pd.Timestamp.now().strftime("%Y-%m-%d"), "n_obs": n,
           "alpha": float(best["alpha"]), "oos_ic": round(float(best["oos_ic_mean"]), 4),
           "variants_tested": len(ALPHAS), "weights": weights, "coef_signs": signs}
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nChosen alpha={best['alpha']}, OOS IC={best['oos_ic_mean']:.3f}")
    print(f"Weights: {weights}")
    neg = [f for f, s in signs.items() if s < 0]
    if neg:
        print(f"WARNING — negative coefficients (factor predicts the WRONG way): {neg}")
        print("Review before activating; a negative-IC factor should be dropped, not weighted.")
    print(f"\nSaved -> {OUT}")
    print('Activate with "factor_weights": "learned" in config.json. Refit quarterly.')


if __name__ == "__main__":
    main()
