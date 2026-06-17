"""
pipeline.py
Daily scoring pipeline: Kronos forecast + 6-factor scoring rubric + regime filter + PyPortfolioOpt.

Scoring factors (equal weight):
  1. Forecast edge       — median Kronos fwd return across sampled paths vs
                           vol-scaled threshold (1-sigma move over horizon)
  2. Path consistency    — P(up): fraction of sampled paths ending positive
  3. Vol context         — RV20 vs 1yr median (penalize high vol)
  4. Trend alignment     — EMA20/EMA50 ratio in 0.5%–3% band
  5. LT quality          — 3yr annualized return proxy (cap 30%)
  6. Gov contract signal — USASpending NEW AWARD / MODIFICATION (defense contractors only)

Modifiers / overlays:
  Congress flow  — net politician buys/sells (30d) nudges score ±congress_modifier
  News sentiment — headline lexicon score, display-only flag (NEWS-RISK / NEWS-POS)
  Paper ledger   — proposals recorded daily, filled at next open, MTM equity curve

Regime multiplier: score × 1.0 if regime_ok, × 0.5 if suppressed
Thresholds: >=0.7 adj -> BUY | 0.4–0.7 -> HOLD | <0.4 -> REDUCE

Usage:
    python pipeline.py                        # score all tickers from config.json
    python pipeline.py --tickers NVDA PLTR    # score specific tickers
    python pipeline.py --no-kronos            # skip Kronos (score on 5 factors only)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = "config.json"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFENSE_CONTRACTORS = {"PLTR", "CACI", "SAIC", "BAH", "LDOS", "RTX", "LMT", "NOC", "GD"}

# trading bars per day by candle interval (US session)
BARS_PER_DAY = {"1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "1d": 1}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def horizon_days(interval: str, pred_len: int) -> float:
    """Forecast horizon in trading days (e.g. 24 hourly candles ~ 3.4 days)."""
    return pred_len / BARS_PER_DAY.get(interval, 7)


def vol_scaled_threshold(hist: pd.DataFrame, interval: str, pred_len: int,
                         mult: float = 1.0, floor: float = 0.005) -> float:
    """
    Edge threshold = mult x expected 1-sigma move over the forecast horizon.
    A fixed 3% hurdle on a ~1-day horizon is nearly unreachable for mega-caps,
    which silently zeroed out the Kronos factor — scale it to the horizon.
    """
    ret = hist["close"].pct_change().dropna()
    if len(ret) < 20:
        return max(floor, 0.03)
    sigma_d = float(ret.iloc[-60:].std())
    sigma_h = sigma_d * np.sqrt(horizon_days(interval, pred_len))
    return max(floor, mult * sigma_h)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df


def macro_event_blackout(days_before: int = 1) -> tuple[bool, str]:
    """True if today is within the blackout window before/on a macro event
    (FOMC/CPI/NFP from macro_calendar.json — editable, verify yearly vs BLS)."""
    cal_path = Path("macro_calendar.json")
    if not cal_path.exists():
        return False, ""
    with open(cal_path) as f:
        cal = json.load(f)
    today = pd.Timestamp.now().normalize()
    for event, dates in cal.items():
        if event.startswith("_"):
            continue
        for d in dates:
            dt = pd.Timestamp(d)
            if 0 <= (dt - today).days <= days_before:
                return True, f"{event} {d}"
    return False, ""


def portfolio_vol_scalar(targets: dict, hist_closes: dict, target_vol: float = 0.10,
                         lo: float = 0.25, hi: float = 1.5) -> tuple[float, float]:
    """
    Gross-exposure scalar = target_vol / realized portfolio vol (20d EWMA,
    annualized), bounded. Returns (scalar, realized_vol).
    """
    total = sum(targets.values())
    if total <= 0:
        return 1.0, 0.0
    rets = pd.DataFrame({t: hist_closes[t].pct_change()
                         for t in targets if t in hist_closes}).dropna()
    if rets.empty or len(rets) < 21:
        return 1.0, 0.0
    w = np.array([targets[t] / total for t in rets.columns])
    port_ret = rets.to_numpy() @ w
    ewma_vol = float(pd.Series(port_ret).ewm(span=20).std().iloc[-1] * np.sqrt(252))
    if ewma_vol <= 0:
        return 1.0, 0.0
    return float(np.clip(target_vol / ewma_vol, lo, hi)), ewma_vol


def next_earnings_in_days(ticker: str) -> float | None:
    """Calendar days until next earnings report, None if unknown/ETF."""
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if ed is None or ed.empty:
            return None
        future = ed.index[ed.index > pd.Timestamp.now(tz=ed.index.tz)]
        if len(future) == 0:
            return None
        return (future.min() - pd.Timestamp.now(tz=ed.index.tz)).total_seconds() / 86400
    except Exception:
        return None


def fetch_fred_series(series_id: str, cache_dir: Path = OUTPUT_DIR / "macro") -> pd.Series:
    """Free FRED CSV endpoint, no API key. Cached daily; 3 attempts then falls
    back to the last-good cache so a transient FRED timeout never silently drops
    the credit/financial-conditions risk gates (observed 2026-06-15)."""
    import requests
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{series_id}_{datetime.now().strftime('%Y%m%d')}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["DATE"])
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        last_err = None
        for _ in range(3):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                df.columns = ["DATE", series_id]
                df.to_csv(cache, index=False)
                df["DATE"] = pd.to_datetime(df["DATE"])
                break
            except Exception as e:
                last_err = e
        else:
            prior = sorted(cache_dir.glob(f"{series_id}_*.csv"))
            if prior:
                print(f"[MACRO] {series_id} live fetch failed ({last_err}); "
                      f"using stale cache {prior[-1].name}")
                df = pd.read_csv(prior[-1], parse_dates=["DATE"])
            else:
                raise last_err
    s = pd.to_numeric(df.set_index("DATE").iloc[:, 0], errors="coerce").dropna()
    return s


def macro_regime(cfg: dict) -> tuple[bool, str]:
    """
    Market-wide credit/conditions gate (one check per run, all tickers):
      HY OAS  > mult x 1y median -> credit stress, suppress
      NFCI    > ceiling          -> tight financial conditions, suppress
    Fetch failure = check skipped (never blocks the pipeline).
    """
    notes = []
    try:
        hy = fetch_fred_series("BAMLH0A0HYM2")
        hy_med = float(hy.iloc[-252:].median())
        hy_now = float(hy.iloc[-1])
        mult = cfg.get("hy_oas_mult", 1.25)
        if hy_now >= mult * hy_med:
            return False, f"HY-OAS {hy_now:.2f}>={mult}x med {hy_med:.2f}"
        notes.append(f"HY {hy_now:.2f}")
    except Exception as e:
        print(f"[MACRO] HY OAS fetch failed: {e}")
    try:
        nfci = fetch_fred_series("NFCI")
        nfci_now = float(nfci.iloc[-1])
        ceiling = cfg.get("nfci_max", 0.0)
        if nfci_now > ceiling:
            return False, f"NFCI {nfci_now:.2f}>{ceiling}"
        notes.append(f"NFCI {nfci_now:.2f}")
    except Exception as e:
        print(f"[MACRO] NFCI fetch failed: {e}")
    return True, " ".join(notes) if notes else "macro-unavailable"


def fetch_vix() -> float:
    vix = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
    col = vix["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return float(col.iloc[-1])


# ---------------------------------------------------------------------------
# Contract signals
# ---------------------------------------------------------------------------

def get_contract_signals(days: int = 7, min_amount: float = 1_000_000) -> dict:
    try:
        import usaspending_watcher
        df = usaspending_watcher.run(days=days, min_amount=min_amount)
        if df.empty or "ticker" not in df.columns or "signal_type" not in df.columns:
            return {}
        signals = {}
        for _, row in df.iterrows():
            t, sig = row["ticker"], row["signal_type"]
            if t not in signals or sig == "NEW AWARD":
                signals[t] = sig
        return signals
    except Exception as e:
        print(f"[CONTRACTS] {e}")
        return {}


# ---------------------------------------------------------------------------
# Scoring factors
# ---------------------------------------------------------------------------

def _forecast_returns(hist: pd.DataFrame, forecast: pd.DataFrame) -> list[float]:
    """Forward return per sampled path; falls back to the single mean path."""
    last = float(hist["close"].iloc[-1])
    paths = forecast.attrs.get("paths")
    if paths:
        return [float(p[-1]) / last - 1.0 for p in paths]
    return [float(forecast["close"].iloc[-1]) / last - 1.0]


def _score_forecast_edge(hist: pd.DataFrame, forecast: pd.DataFrame, threshold: float) -> float:
    rets = _forecast_returns(hist, forecast)
    med = float(np.median(rets))
    return min(1.0, max(0.0, med / threshold))


def _score_path_consistency(hist: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """
    With sampled paths: P(up) = fraction of paths ending positive — a real
    probability estimate. Single path: fraction of up-steps (legacy).
    """
    paths = forecast.attrs.get("paths")
    if paths:
        rets = _forecast_returns(hist, forecast)
        return sum(1 for r in rets if r > 0) / len(rets)
    c = forecast["close"].values
    up = sum(1 for i in range(1, len(c)) if c[i] > c[i - 1])
    return up / max(1, len(c) - 1)


def _score_vol_context(hist: pd.DataFrame) -> float:
    ret = hist["close"].pct_change().dropna()
    if len(ret) < 20:
        return 0.5
    rv20 = ret.iloc[-20:].std() * np.sqrt(252)
    rv_med = ret.rolling(252).std().dropna().median() * np.sqrt(252)
    if rv_med == 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (rv20 / rv_med - 1.0)))


def _score_trend_alignment(hist: pd.DataFrame) -> float:
    c = hist["close"]
    if len(c) < 50:
        return 0.5
    ema20 = c.ewm(span=20).mean().iloc[-1]
    ema50 = c.ewm(span=50).mean().iloc[-1]
    ratio = abs(ema20 / ema50 - 1)
    if 0.005 <= ratio <= 0.03:
        return 1.0
    if ratio < 0.005:
        return ratio / 0.005
    return max(0.0, 1.0 - (ratio - 0.03) / 0.03)


def _score_lt_quality(hist: pd.DataFrame) -> float:
    c = hist["close"].dropna()
    if len(c) < 252:
        return 0.5
    years = len(c) / 252
    ann = (float(c.iloc[-1]) / float(c.iloc[0])) ** (1 / years) - 1
    return min(1.0, max(0.0, ann / 0.30))


def _score_contract(ticker: str, signals: dict) -> float:
    sig = signals.get(ticker)
    if sig == "NEW AWARD":
        return 1.0
    if sig == "MODIFICATION":
        return 0.6
    if ticker in DEFENSE_CONTRACTORS:
        return 0.3
    return 0.0


def compute_score(
    hist: pd.DataFrame,
    forecast: pd.DataFrame | None,
    ticker: str,
    contract_signals: dict,
    threshold: float = 0.03,
    weights: dict | None = None,
) -> dict:
    f3 = _score_vol_context(hist)
    f4 = _score_trend_alignment(hist)
    f5 = _score_lt_quality(hist)
    f6 = _score_contract(ticker, contract_signals)

    kronos_fwd_ret = None
    if forecast is not None:
        f1 = _score_forecast_edge(hist, forecast, threshold)
        f2 = _score_path_consistency(hist, forecast)
        kronos_fwd_ret = float(np.median(_forecast_returns(hist, forecast)))
    else:
        f1 = f2 = 0.5  # neutral when Kronos skipped

    is_defense = ticker in DEFENSE_CONTRACTORS
    fvals = {"forecast_edge": f1, "path_consistency": f2, "vol_context": f3,
             "trend_alignment": f4, "lt_quality": f5}
    if is_defense:
        fvals["contract_signal_score"] = f6
    if weights:
        used = {k: weights[k] for k in fvals if weights.get(k, 0) > 0}
        if used:
            wsum = sum(used.values())
            raw = sum(fvals[k] * w for k, w in used.items()) / wsum
        else:
            raw = sum(fvals.values()) / len(fvals)
    else:
        raw = sum(fvals.values()) / len(fvals)

    return {
        "forecast_edge": round(f1, 3),
        "path_consistency": round(f2, 3),
        "vol_context": round(f3, 3),
        "trend_alignment": round(f4, 3),
        "lt_quality": round(f5, 3),
        "contract_signal_score": round(f6, 3),
        "raw_score": round(raw, 3),
        "kronos_fwd_ret": kronos_fwd_ret,
        "edge_threshold": round(threshold, 4),
    }


# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------

def check_regime(hist: pd.DataFrame, vix: float) -> tuple[bool, str]:
    if vix >= 22:
        return False, f"VIX={vix:.1f}>=22"

    ret = hist["close"].pct_change().dropna()
    if len(ret) >= 20:
        rv20 = ret.iloc[-20:].std() * np.sqrt(252)
        rv_med = ret.rolling(252).std().dropna().median() * np.sqrt(252)
        if rv_med > 0 and rv20 >= 1.2 * rv_med:
            return False, f"RV20={rv20:.3f}>=1.2×med({rv_med:.3f})"

    c = hist["close"]
    if len(c) >= 50:
        ema20 = c.ewm(span=20).mean().iloc[-1]
        ema50 = c.ewm(span=50).mean().iloc[-1]
        if abs(ema20 / ema50 - 1) < 0.01:
            return False, "EMA20/EMA50 flat"

    return True, "ok"


def action_label(raw_score: float, regime: bool) -> str:
    """
    De-conflicted risk stack: regime is an ENTRY GATE only (bad regime blocks
    new BUYs, never force-halves the score), the score ranks, and portfolio
    vol targeting handles sizing. Halving scores on top of gating on top of
    vol scaling triple-counts risk-off and over-de-risks.
    """
    # REDUCE cutoff 0.45 (was 0.40): 2026-06-15 factor-mode sweep showed hold_thr
    # 0.45 lifts portfolio Sharpe ~0.55->0.71 and trims max drawdown vs 0.40,
    # robustly across the grid. BUY cutoff 0.70 unchanged (0.65-0.75 ~equivalent).
    # See notes/hermes_notes/2026-06-15-CARRY-FORWARD.md
    if raw_score < 0.45:
        return "REDUCE"
    if raw_score >= 0.7 and regime:
        return "BUY"
    return "HOLD"


# ---------------------------------------------------------------------------
# Portfolio optimization
# ---------------------------------------------------------------------------

def annualize_kronos_mu(fwd_ret: float, h_days: float, cap: float = 0.60) -> float:
    """
    Annualize a Kronos horizon return so it shares units with
    mean_historical_return (annualized). Clipped — compounding a short-horizon
    return to a year explodes, the cap keeps the optimizer sane.
    """
    if h_days <= 0:
        return 0.0
    ann = (1.0 + fwd_ret) ** (252.0 / h_days) - 1.0
    return float(np.clip(ann, -cap, cap))


def optimize_portfolio(
    scores: dict,
    regimes: dict,
    hist_closes: dict,
    portfolio_value: float,
    max_position_pct: float = 0.20,
    h_days: float = 1.0,
    mu_blend: float = 0.5,
    mu_cap: float = 0.60,
) -> dict:
    eligible = {t for t in scores if action_label(scores[t]["raw_score"], regimes[t]) != "REDUCE"}
    if not eligible:
        return {}

    # fallback weight respects the position cap (rest stays cash)
    eq_weight = round(min(1 / len(eligible), max_position_pct), 4)

    try:
        from pypfopt import EfficientFrontier, expected_returns, risk_models

        if len(eligible) < 2:
            return {t: round(eq_weight * portfolio_value, 2) for t in eligible}

        prices = pd.DataFrame({t: hist_closes[t] for t in eligible}).dropna()
        if prices.shape[0] < 60:
            raise ValueError("insufficient history for optimizer")

        mu = expected_returns.mean_historical_return(prices)
        for t in eligible:
            fwd = scores[t].get("kronos_fwd_ret")
            if fwd is not None:
                kronos_ann = annualize_kronos_mu(fwd, h_days, cap=mu_cap)
                mu[t] = (1 - mu_blend) * float(mu[t]) + mu_blend * kronos_ann

        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, max_position_pct))
        ef.max_sharpe()
        cw = ef.clean_weights()
        return {t: round(w * portfolio_value, 2) for t, w in cw.items() if w > 0}

    except Exception as e:
        print(f"[OPTIMIZER] {e} — equal weight fallback")
        return {t: round(eq_weight * portfolio_value, 2) for t in eligible}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(tickers: list | None = None, use_kronos: bool = True) -> pd.DataFrame:
    cfg = load_config()
    all_tickers = tickers or cfg["tickers"]
    forecast_tickers = [t for t in all_tickers if t not in ("SPY", "QQQ")]

    if use_kronos:
        from kronos_forecast import run_forecast

    print(f"\n{'='*65}")
    print(f"PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Tickers: {forecast_tickers}  |  Kronos: {'on' if use_kronos else 'off'}")
    print(f"{'='*65}\n")

    # Paper ledger: fill yesterday's pending proposals at today's open, MTM
    if cfg.get("paper_trading", False):
        try:
            import ledger
            ledger.mark(starting_cash=cfg.get("paper_cash_start", cfg["portfolio_value"]),
                        slippage_bps=cfg.get("slippage_bps", 10.0))
        except Exception as e:
            print(f"[LEDGER] mark failed: {e}")

    print("Fetching VIX...")
    vix = fetch_vix()
    print(f"VIX: {vix:.1f}")

    print("Checking macro regime (FRED)...")
    macro_ok, macro_note = macro_regime(cfg)
    print(f"Macro: {'OK' if macro_ok else 'SUPPRESS'} ({macro_note})")

    print("\nFetching gov contract signals (7-day window)...")
    contract_signals = get_contract_signals(days=7, min_amount=1_000_000)
    print(f"Signals: {contract_signals if contract_signals else 'none'}")

    congress = {}
    if cfg.get("congress_enabled", True):
        print("\nFetching congressional trades...")
        try:
            from quiver_congress_watchlist import get_signals
            congress = get_signals(forecast_tickers, days=cfg.get("congress_days", 30))
            print(f"Congress activity: {congress if congress else 'none'}")
        except Exception as e:
            print(f"[CONGRESS] {e}")

    news = {}
    if cfg.get("news_enabled", True):
        print("\nFetching headlines...")
        try:
            from news_watcher import get_news_signals
            news = get_news_signals(forecast_tickers,
                                    use_finbert=cfg.get("news_model", "finbert") == "finbert")
        except Exception as e:
            print(f"[NEWS] {e}")

    insiders = {}
    if cfg.get("insider_enabled", True):
        print("\nFetching SEC Form 4 insider buys...")
        try:
            from edgar_watcher import get_insider_signals
            insiders = get_insider_signals(forecast_tickers)
            print(f"Insider clusters: {insiders if insiders else 'none'}")
        except Exception as e:
            print(f"[EDGAR] {e}")

    event_blackout, event_note = macro_event_blackout(cfg.get("macro_blackout_days", 1))
    if event_blackout:
        print(f"\nMACRO EVENT BLACKOUT: {event_note} — new entries vetoed today")

    weights = None
    if cfg.get("factor_weights") == "learned":
        wpath = Path("output/learned_weights.json")
        if wpath.exists():
            with open(wpath) as f:
                weights = json.load(f).get("weights")
            print(f"Using learned factor weights: {weights}")
        else:
            print("[WEIGHTS] learned_weights.json missing — equal weights")

    scores, regimes, regime_notes, hist_closes = {}, {}, {}, {}
    cong_mod_size = cfg.get("congress_modifier", 0.05)

    for ticker in forecast_tickers:
        print(f"\n[{ticker}]")
        try:
            hist = fetch_history(ticker)
            if hist.empty:
                print("  no history — skip")
                continue

            forecast = None
            if use_kronos:
                forecast = run_forecast(
                    ticker=ticker,
                    interval=cfg["interval"],
                    pred_len=cfg["pred_len"],
                    model_key=cfg["model"],
                    out_dir="output",
                    num_paths=cfg.get("num_paths", 3),
                    reuse_within_hours=cfg.get("forecast_reuse_hours", 0),
                    make_plot=cfg.get("make_plots", True),
                )

            threshold = vol_scaled_threshold(
                hist, cfg["interval"], cfg["pred_len"],
                mult=cfg.get("edge_vol_mult", 1.0),
                floor=cfg.get("edge_floor", 0.005),
            )
            s = compute_score(hist, forecast, ticker, contract_signals, threshold,
                              weights=weights)

            # Bounded modifiers: congress flow + insider cluster buys
            c = congress.get(ticker)
            cong_mod = 0.0
            if c:
                if c["buys"] > c["sells"]:
                    cong_mod = cong_mod_size
                elif c["sells"] > c["buys"]:
                    cong_mod = -cong_mod_size
            ins = insiders.get(ticker)
            ins_mod = {"STRONG": 0.10, "WEAK": 0.05}.get(ins["strength"], 0.0) if ins else 0.0
            s["congress_mod"] = cong_mod
            s["insider_mod"] = ins_mod
            s["raw_score"] = round(float(np.clip(s["raw_score"] + cong_mod + ins_mod, 0.0, 1.0)), 3)

            ok, note = check_regime(hist, vix)
            if ok and not macro_ok:
                ok, note = False, macro_note

            # Earnings blackout: foundation forecasters can't see gap events
            # coming — veto NEW entries just before a report
            days_to_er = next_earnings_in_days(ticker)
            blackout = (days_to_er is not None
                        and 0 <= days_to_er <= cfg.get("earnings_blackout_days", 3))
            s["earnings_blackout"] = blackout
            s["days_to_earnings"] = round(days_to_er, 1) if days_to_er is not None else None
            act = action_label(s["raw_score"], ok)

            scores[ticker] = s
            regimes[ticker] = ok
            regime_notes[ticker] = note
            hist_closes[ticker] = hist["close"].dropna()

            print(f"  score={s['raw_score']:.3f}  regime={'OK' if ok else 'GATE:'+note}  -> {act}")

        except Exception as e:
            print(f"  ERROR: {e}")

    if not scores:
        print("\nNo tickers scored.")
        return pd.DataFrame()

    print("\nOptimizing portfolio weights...")
    targets = optimize_portfolio(
        scores, regimes, hist_closes,
        portfolio_value=cfg["portfolio_value"],
        max_position_pct=cfg["max_position_pct"],
        h_days=horizon_days(cfg["interval"], cfg["pred_len"]),
        mu_blend=cfg.get("mu_blend", 0.5),
        mu_cap=cfg.get("mu_cap", 0.60),
    )

    # Vol targeting: scale gross exposure to the portfolio vol target.
    # This is the only risk-off sizing lever (regime gates entries, score ranks).
    scalar, realized_vol = portfolio_vol_scalar(
        targets, hist_closes,
        target_vol=cfg.get("target_vol", 0.10),
        lo=cfg.get("vol_scalar_min", 0.25),
        hi=cfg.get("vol_scalar_max", 1.5),
    )
    if realized_vol > 0:
        targets = {t: round(v * scalar, 2) for t, v in targets.items()}
        print(f"Vol targeting: realized {realized_vol*100:.1f}% vs target "
              f"{cfg.get('target_vol', 0.10)*100:.0f}% -> gross scalar {scalar:.2f}x")

    rows = []
    for ticker in forecast_tickers:
        if ticker not in scores:
            continue
        s = scores[ticker]
        ok = regimes[ticker]
        n = news.get(ticker, {})
        c = congress.get(ticker, {})
        act = action_label(s["raw_score"], ok)
        if act == "BUY" and s.get("earnings_blackout"):
            act = "HOLD"  # entry vetoed: earnings within blackout window
        if act == "BUY" and event_blackout:
            act = "HOLD"  # entry vetoed: FOMC/CPI/NFP window
        if act == "BUY" and n.get("veto"):
            act = "HOLD"  # entry vetoed: adverse news event class
        rows.append({
            "ticker": ticker,
            "score": s["raw_score"],
            "adj_score": s["raw_score"],
            "action": act,
            "er_blackout": bool(s.get("earnings_blackout")),
            "macro_event": event_note if event_blackout else "-",
            "news_veto": n.get("veto_reason", "") or "-",
            "days_to_earnings": s.get("days_to_earnings"),
            "insider": (f"{insiders[ticker]['buyers']}buy/${insiders[ticker]['dollars']/1000:.0f}k"
                        if ticker in insiders else "-"),
            "insider_mod": s.get("insider_mod", 0.0),
            "regime_ok": ok,
            "regime_note": regime_notes[ticker],
            "contract_signal": contract_signals.get(ticker, "-"),
            "congress": f"{c.get('buys', 0)}B/{c.get('sells', 0)}S" if c else "-",
            "news_flag": n.get("flag", "-"),
            "news_sent": n.get("sent", 0.0),
            "forecast_edge": s["forecast_edge"],
            "path_consistency": s["path_consistency"],
            "vol_context": s["vol_context"],
            "trend_alignment": s["trend_alignment"],
            "lt_quality": s["lt_quality"],
            "contract_signal_score": s["contract_signal_score"],
            "congress_mod": s.get("congress_mod", 0.0),
            "kronos_fwd_ret": s.get("kronos_fwd_ret"),
            "edge_threshold": s.get("edge_threshold"),
            "target_value": targets.get(ticker, 0.0),
        })

    df = pd.DataFrame(rows).sort_values("adj_score", ascending=False)

    print(f"\n{'='*65}")
    print("PROPOSAL TABLE")
    print(f"{'='*65}")
    print(df[["ticker", "score", "action", "regime_ok", "er_blackout",
              "contract_signal", "congress", "insider", "news_flag",
              "target_value"]].to_string(index=False))
    print(f"\nPortfolio value: ${cfg['portfolio_value']:,.0f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"proposals_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")

    # Append-only decision log: factor values + price at decision time.
    # ic_report.py later joins realized forward returns onto this — the
    # training data for learned factor weights. Never delete this file.
    log_df = df.copy()
    log_df.insert(0, "run_date", datetime.now().strftime("%Y-%m-%d"))
    log_df["close_at_score"] = [
        round(float(hist_closes[t].iloc[-1]), 2) if t in hist_closes else None
        for t in log_df["ticker"]
    ]
    log_df["vix"] = vix
    factor_log = OUTPUT_DIR / "factor_log.csv"
    if factor_log.exists():
        # schema drift guard: appending different columns under an old header
        # silently corrupts the file (June 2026 lesson) — align before append
        with open(factor_log) as f:
            existing_cols = f.readline().strip().split(",")
        if existing_cols != list(log_df.columns):
            old = pd.read_csv(factor_log)
            merged = pd.concat([old, log_df], ignore_index=True)
            merged.to_csv(factor_log, index=False)
            print(f"Decision log schema migrated + appended -> {factor_log}")
        else:
            log_df.to_csv(factor_log, mode="a", header=False, index=False)
            print(f"Decision log appended -> {factor_log}")
    else:
        log_df.to_csv(factor_log, mode="a", header=True, index=False)
        print(f"Decision log appended -> {factor_log}")

    if cfg.get("paper_trading", False):
        try:
            import ledger
            ledger.record(df)
        except Exception as e:
            print(f"[LEDGER] record failed: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Quant scoring pipeline")
    parser.add_argument("--tickers", nargs="+", help="Override ticker list")
    parser.add_argument("--no-kronos", action="store_true", help="Skip Kronos forecasts (5-factor mode)")
    args = parser.parse_args()
    run_pipeline(tickers=args.tickers, use_kronos=not args.no_kronos)


if __name__ == "__main__":
    main()
