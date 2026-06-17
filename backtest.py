"""
backtest.py
Validates the scoring rubric before real money arrives. Two modes, no new
dependencies (pandas-native — vectorbt is incompatible with pandas 3.x).

MODE A — factors (fast, default)
    Weekly walk-forward of the non-Kronos factors (vol context, trend
    alignment, LT quality) + regime filter over daily history. No look-ahead:
    every input at week t uses data up to t only. Reports CAGR / Sharpe /
    max drawdown / exposure vs buy-and-hold, per ticker and blended.

    python backtest.py                          # all config tickers, 3y
    python backtest.py --tickers NVDA META      # subset
    python backtest.py --sweep                  # grid-search BUY/HOLD cutoffs

MODE B — kronos (slow, real inference)
    Walk-forward Kronos forecasts at weekly as-of dates (hourly data, no
    look-ahead), compared against realized forward returns. Reports IC
    (Spearman rank correlation between forecast and realized), hit rate,
    and average realized return when the forecast cleared its threshold.
    Each run is one CPU inference — cost is printed up front and any job
    over 5 runs requires --yes.

    python backtest.py --mode kronos --tickers NVDA CACI --weeks 8 --yes
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from pipeline import BARS_PER_DAY, horizon_days, vol_scaled_threshold

CONFIG_PATH = "config.json"
OUTPUT_DIR = Path("output")
TRADING_WEEK = 52


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared metrics
# ---------------------------------------------------------------------------

def perf_stats(weekly_ret: pd.Series) -> dict:
    r = weekly_ret.dropna()
    if r.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_WEEK
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_WEEK)) if r.std() > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": dd}


# ---------------------------------------------------------------------------
# MODE A — factor backtest (vectorized, no Kronos)
# ---------------------------------------------------------------------------

def factor_series(close: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Daily factor scores + regime flag, each value using data up to that day."""
    ret = close.pct_change()

    rv20 = ret.rolling(20).std() * np.sqrt(252)
    rv252 = ret.rolling(252).std() * np.sqrt(252)
    rv_med = rv252.expanding().median()
    vol_score = (1.0 - (rv20 / rv_med - 1.0)).clip(0.0, 1.0)

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    ratio = (ema20 / ema50 - 1.0).abs()
    trend_score = pd.Series(
        np.select(
            [(ratio >= 0.005) & (ratio <= 0.03), ratio < 0.005],
            [1.0, ratio / 0.005],
            default=(1.0 - (ratio - 0.03) / 0.03),
        ),
        index=close.index,
    ).clip(0.0, 1.0)

    n = pd.Series(np.arange(1, len(close) + 1), index=close.index, dtype=float)
    ann = (close / close.iloc[0]) ** (252.0 / n) - 1.0
    ltq_score = (ann / 0.30).clip(0.0, 1.0)
    ltq_score[n < 252] = 0.5

    vix_aligned = vix.reindex(close.index).ffill()
    regime = (vix_aligned < 22) & (rv20 < 1.2 * rv_med) & (ratio >= 0.01)

    out = pd.DataFrame({
        "raw": (vol_score + trend_score + ltq_score) / 3,
        "regime": regime.fillna(False),
        "close": close,
    })
    return out


def backtest_factor_ticker(fs: pd.DataFrame, buy_thr: float, hold_thr: float) -> pd.DataFrame:
    """
    Weekly position state machine, gate semantics matching live pipeline:
    entry requires score >= buy_thr AND regime ok; exit on score < hold_thr.
    Regime never halves scores — it only blocks new entries.
    """
    wk = fs.resample("W-FRI").last().dropna(subset=["close"])
    wk["ret"] = wk["close"].pct_change()

    pos = []
    holding = 0
    for raw, regime in zip(wk["raw"], wk["regime"]):
        if holding == 0 and raw >= buy_thr and regime:
            holding = 1
        elif holding == 1 and raw < hold_thr:
            holding = 0
        pos.append(holding)
    wk["pos"] = pos
    wk["strat_ret"] = wk["pos"].shift(1).fillna(0) * wk["ret"]
    return wk


def run_factor_mode(tickers: list, period: str, buy_thr: float, hold_thr: float,
                    sweep: bool = False):
    # warmup: rolling-252 needs a year of data before signals fire
    print(f"Downloading daily history ({period}) for {len(tickers)} tickers + ^VIX...")
    data = yf.download(tickers + ["^VIX"], period=period, interval="1d",
                       progress=False, auto_adjust=True, group_by="ticker")
    vix = data["^VIX"]["Close"].dropna()

    all_fs = {}
    for t in tickers:
        try:
            close = data[t]["Close"].dropna()
            if len(close) < 300:
                print(f"  {t}: insufficient history ({len(close)} days) — skip")
                continue
            all_fs[t] = factor_series(close, vix)
        except Exception as e:
            print(f"  {t}: {e}")

    if not all_fs:
        print("Nothing to backtest.")
        return

    if sweep:
        print("\nGrid search (portfolio Sharpe across tickers):")
        grid_buy = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        grid_hold = [0.30, 0.35, 0.40, 0.45]
        results = []
        for b in grid_buy:
            for h in grid_hold:
                if h >= b:
                    continue
                rets = pd.DataFrame({
                    t: backtest_factor_ticker(fs, b, h)["strat_ret"]
                    for t, fs in all_fs.items()
                }).mean(axis=1)
                st = perf_stats(rets)
                results.append({"buy_thr": b, "hold_thr": h, **st})
        rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        rdf[["cagr", "max_dd"]] = (rdf[["cagr", "max_dd"]] * 100).round(1)
        rdf["sharpe"] = rdf["sharpe"].round(2)
        print(rdf.to_string(index=False))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"backtest_sweep_{ts}.csv"
        rdf.to_csv(out, index=False)
        print(f"\nSaved -> {out}")
        return

    rows, strat_rets, bh_rets = [], {}, {}
    for t, fs in all_fs.items():
        wk = backtest_factor_ticker(fs, buy_thr, hold_thr)
        st = perf_stats(wk["strat_ret"])
        bh = perf_stats(wk["ret"])
        strat_rets[t] = wk["strat_ret"]
        bh_rets[t] = wk["ret"]
        rows.append({
            "ticker": t,
            "strat_cagr%": round(st["cagr"] * 100, 1),
            "bh_cagr%": round(bh["cagr"] * 100, 1),
            "strat_sharpe": round(st["sharpe"], 2),
            "bh_sharpe": round(bh["sharpe"], 2),
            "strat_maxdd%": round(st["max_dd"] * 100, 1),
            "bh_maxdd%": round(bh["max_dd"] * 100, 1),
            "exposure%": round(wk["pos"].mean() * 100, 0),
        })

    df = pd.DataFrame(rows)
    port = perf_stats(pd.DataFrame(strat_rets).mean(axis=1))
    port_bh = perf_stats(pd.DataFrame(bh_rets).mean(axis=1))

    print(f"\nFactor backtest (BUY>={buy_thr}, exit<{hold_thr}, weekly, {period}):\n")
    print(df.to_string(index=False))
    print(f"\nPORTFOLIO (equal-weight): "
          f"strat CAGR {port['cagr']*100:.1f}% Sharpe {port['sharpe']:.2f} "
          f"maxDD {port['max_dd']*100:.1f}%  |  "
          f"buy-hold CAGR {port_bh['cagr']*100:.1f}% Sharpe {port_bh['sharpe']:.2f} "
          f"maxDD {port_bh['max_dd']*100:.1f}%")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"backtest_factors_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# MODE B — Kronos walk-forward (real inference, costed)
# ---------------------------------------------------------------------------

def run_kronos_mode(tickers: list, weeks: int, cfg: dict, confirmed: bool):
    from kronos_data_fetcher import fetch_ohlcv
    from kronos_forecast import run_forecast

    pred_len = cfg["pred_len"]
    interval = cfg["interval"]
    runs = len(tickers) * weeks
    print(f"Kronos walk-forward: {len(tickers)} tickers x {weeks} weekly as-of dates "
          f"= {runs} inference runs (~30-90s each on CPU).")
    if runs > 5 and not confirmed:
        print("More than 5 runs — re-run with --yes to confirm.")
        return

    pairs = []
    for t in tickers:
        print(f"\n[{t}] fetching full hourly history...")
        try:
            full = fetch_ohlcv(t, interval, lookback=10**6)
        except Exception as e:
            print(f"  {e}")
            continue

        # weekly as-of dates: last bar of each week, leaving room for the
        # realized forward window after the final as-of date
        bars = full["timestamps"]
        weekly_last = full.groupby(bars.dt.to_period("W"))["timestamps"].max()
        asofs = weekly_last.iloc[-(weeks + 1):-1]  # most recent N full weeks

        for asof in asofs:
            try:
                pred = run_forecast(
                    ticker=t, interval=interval, pred_len=pred_len,
                    lookback=cfg["lookback"], model_key=cfg["model"],
                    num_paths=cfg.get("num_paths", 3),
                    asof=asof, prefetched_df=full, make_plot=False,
                )
                i = full.index[bars <= asof][-1]
                if i + pred_len >= len(full):
                    continue
                last = float(full["close"].iloc[i])
                realized = float(full["close"].iloc[i + pred_len]) / last - 1.0
                paths = pred.attrs.get("paths")
                fwd = (float(np.median([p[-1] for p in paths])) / last - 1.0) if paths \
                    else (float(pred["close"].iloc[-1]) / last - 1.0)
                hist_slice = full.iloc[: i + 1].rename(columns={"close": "close"})
                thr = vol_scaled_threshold(hist_slice, interval, pred_len,
                                           cfg.get("edge_vol_mult", 1.0),
                                           cfg.get("edge_floor", 0.005))
                pairs.append({"ticker": t, "asof": asof, "pred_ret": fwd,
                              "realized_ret": realized, "threshold": thr,
                              "signal": fwd >= thr})
                print(f"  {asof}  pred={fwd:+.4f}  realized={realized:+.4f}  "
                      f"thr={thr:.4f}  {'SIGNAL' if fwd >= thr else ''}")
            except Exception as e:
                print(f"  {asof}: {e}")

    if not pairs:
        print("No forecast/realized pairs produced.")
        return

    df = pd.DataFrame(pairs)
    try:
        from scipy.stats import spearmanr
        ic = float(spearmanr(df["pred_ret"], df["realized_ret"]).statistic)
    except Exception:
        ic = float(df["pred_ret"].rank().corr(df["realized_ret"].rank()))

    hit = float((np.sign(df["pred_ret"]) == np.sign(df["realized_ret"])).mean())
    sig = df[df["signal"]]
    print(f"\n{'='*60}\nKRONOS WALK-FORWARD RESULTS  ({len(df)} pairs)\n{'='*60}")
    print(f"IC (Spearman pred vs realized): {ic:+.3f}   (>0.05 = usable edge)")
    print(f"Direction hit rate:             {hit*100:.0f}%   (>50% = better than coin)")
    print(f"Avg realized, all pairs:        {df['realized_ret'].mean()*100:+.2f}%")
    if not sig.empty:
        print(f"Avg realized, SIGNAL fired:     {sig['realized_ret'].mean()*100:+.2f}%  "
              f"({len(sig)} signals)")
    else:
        print("No signals cleared the threshold in this window.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"backtest_kronos_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline backtester")
    parser.add_argument("--mode", choices=["factors", "kronos"], default="factors")
    parser.add_argument("--tickers", nargs="+", help="Override config tickers")
    parser.add_argument("--period", default="3y", help="Daily history span (factors mode)")
    parser.add_argument("--buy-thr", type=float, default=0.7)
    parser.add_argument("--hold-thr", type=float, default=0.4)
    parser.add_argument("--sweep", action="store_true", help="Grid-search thresholds (factors mode)")
    parser.add_argument("--weeks", type=int, default=8, help="As-of dates per ticker (kronos mode)")
    parser.add_argument("--yes", action="store_true", help="Confirm kronos runs > 5")
    parser.add_argument("--interval", help="Override cfg interval for kronos mode, e.g. 1d")
    parser.add_argument("--horizon", type=int, help="Override cfg pred_len (forecast bars) for kronos mode")
    args = parser.parse_args()

    cfg = load_config()
    if args.interval:
        cfg["interval"] = args.interval
    if args.horizon:
        cfg["pred_len"] = args.horizon
    tickers = args.tickers or [t for t in cfg["tickers"] if t not in ("SPY", "QQQ")]
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.mode == "factors":
        run_factor_mode(tickers, args.period, args.buy_thr, args.hold_thr, sweep=args.sweep)
    else:
        run_kronos_mode(tickers, args.weeks, cfg, confirmed=args.yes)


if __name__ == "__main__":
    main()
