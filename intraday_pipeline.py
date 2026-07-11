"""
intraday_pipeline.py
Intraday scoring: 4 pure-technical factors. No Kronos (1h IC = -0.130, unusable).

Factors (equal weight):
  1. momentum_signal  — 7-bar price change (≈1 trading day) normalized to [0,1]
  2. volume_surge     — current bar volume vs 20-bar rolling avg
  3. trend_alignment  — EMA20/EMA50 ratio, same as daily pipeline
  4. vol_context      — RV20 vs 26-day median (penalize choppy tape)

Gates (precedence order):
  1. Time gate        — no entries before 09:45 ET; force-close ≥15:25 ET
  2. PDT budget       — ≤3 round-trips per 7 calendar days (≈5 trading days)
  3. Macro event      — FOMC/CPI/NFP blackout via macro_calendar.json
  4. VIX ceiling      — no entries when VIX > 30
  5. Daily gate       — skip if daily pipeline scored REDUCE for that ticker
  6. News veto        — hard veto via news_watcher.py (7 categories)
  7. Max positions    — 2 simultaneous cap

Auto-detects force-close mode when called at or after force_close_at (15:25).

Usage:
    python intraday_pipeline.py          # scan (auto time-detect)
    python intraday_pipeline.py --scan   # force scan
    python intraday_pipeline.py --force-close
"""

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = Path("intraday_config.json")
OUTPUT_DIR  = Path("output/intraday")
LEDGER_DIR  = Path("ledger/intraday")

BARS_PER_DAY_1H = 6.5  # US session hours
ANNUALIZE_1H    = np.sqrt(252 * BARS_PER_DAY_1H)  # ~40.5


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def now_et() -> datetime:
    # Uses local clock; machine must be set to US/Eastern.
    return datetime.now().astimezone()


def time_et_str() -> str:
    return now_et().strftime("%H:%M")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_1h_bulk(tickers: list, lookback: int = 168) -> dict[str, pd.DataFrame]:
    """Single yfinance call for all tickers — avoids per-ticker rate limits."""
    raw = yf.download(tickers, period="30d", interval="1h",
                      auto_adjust=True, progress=False, group_by="ticker")
    if raw.empty:
        return {}
    result = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                sub = raw
            else:
                sub = raw[ticker]
            if isinstance(sub.columns, pd.MultiIndex):
                sub = sub.droplevel(0, axis=1)
            df = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "timestamps"
            df = df.reset_index()
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            df = df.dropna().tail(lookback).reset_index(drop=True)
            if not df.empty:
                result[ticker] = df
        except Exception as e:
            print(f"  [{ticker}] parse error: {e}")
    return result


def fetch_vix() -> float:
    v = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
    col = v["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return float(col.iloc[-1])


def fetch_last_price(ticker: str) -> float | None:
    try:
        q = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        return float(q["Close"].iloc[-1]) if not q.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring factors
# ---------------------------------------------------------------------------

def _score_momentum(df: pd.DataFrame, lookback_bars: int = 7) -> float:
    if len(df) < lookback_bars + 1:
        return 0.5
    past = float(df["close"].iloc[-(lookback_bars + 1)])
    now  = float(df["close"].iloc[-1])
    pct  = (now - past) / past
    # ±3% maps to 0/1; flat = 0.5
    return float(np.clip(0.5 + pct / 0.06, 0.0, 1.0))


def _score_volume_surge(df: pd.DataFrame, window: int = 20) -> float:
    if len(df) < window + 1:
        return 0.5
    avg = float(df["volume"].iloc[-window - 1:-1].mean())
    if avg <= 0:
        return 0.5
    ratio = float(df["volume"].iloc[-1]) / avg
    # 1.0x avg = 0.0; 1.6x avg = 1.0
    return float(np.clip((ratio - 1.0) / 0.6, 0.0, 1.0))


def _score_trend_alignment(df: pd.DataFrame) -> float:
    c = df["close"]
    if len(c) < 50:
        return 0.5
    ema20 = float(c.ewm(span=20).mean().iloc[-1])
    ema50 = float(c.ewm(span=50).mean().iloc[-1])
    ratio = abs(ema20 / ema50 - 1)
    if 0.005 <= ratio <= 0.03:
        return 1.0
    if ratio < 0.005:
        return ratio / 0.005
    return max(0.0, 1.0 - (ratio - 0.03) / 0.03)


def _score_vol_context(df: pd.DataFrame) -> float:
    ret = df["close"].pct_change().dropna()
    if len(ret) < 20:
        return 0.5
    rv20    = ret.iloc[-20:].std() * ANNUALIZE_1H
    # baseline window must fit inside the 168-bar lookback (half, not full 168)
    rv_med  = ret.rolling(84).std().dropna().median() * ANNUALIZE_1H
    if not np.isfinite(rv_med) or rv_med <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (rv20 / rv_med - 1.0)))


def compute_score(df: pd.DataFrame, cfg: dict) -> dict:
    f1 = _score_momentum(df, cfg.get("momentum_lookback_bars", 7))
    f2 = _score_volume_surge(df)
    f3 = _score_trend_alignment(df)
    f4 = _score_vol_context(df)
    return {
        "momentum_signal":  round(f1, 3),
        "volume_surge":     round(f2, 3),
        "trend_alignment":  round(f3, 3),
        "vol_context":      round(f4, 3),
        "score":            round((f1 + f2 + f3 + f4) / 4.0, 3),
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def macro_event_blackout(days_before: int = 1) -> tuple[bool, str]:
    cal = Path("macro_calendar.json")
    if not cal.exists():
        return False, ""
    with open(cal) as f:
        events = json.load(f)
    today = pd.Timestamp.now().normalize()
    for event, dates in events.items():
        if event.startswith("_"):
            continue
        for d in dates:
            dt = pd.Timestamp(d)
            if 0 <= (dt - today).days <= days_before:
                return True, f"{event} {d}"
    return False, ""


def load_pdt_tracker() -> dict:
    path = LEDGER_DIR / "pdt_tracker.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"trades": []}


def pdt_count(tracker: dict, rolling_days: int = 7) -> int:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=rolling_days)
    return sum(1 for t in tracker["trades"] if pd.Timestamp(t) >= cutoff)


def save_pdt_tracker(tracker: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_DIR / "pdt_tracker.json", "w") as f:
        json.dump(tracker, f, indent=2)


def load_open_positions() -> pd.DataFrame:
    path = LEDGER_DIR / "positions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["ticker", "entry_price", "qty", "entry_time", "stop", "target"])


def save_open_positions(df: pd.DataFrame) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEDGER_DIR / "positions.csv", index=False)


def daily_gate_action(ticker: str) -> str:
    """Read latest daily proposals CSV; return action or HOLD if unknown."""
    candidates = sorted(p for p in Path("output").glob("proposals_*.csv")
                        if "intraday" not in p.name)
    if not candidates:
        return "HOLD"
    df = pd.read_csv(candidates[-1])
    row = df[df["ticker"] == ticker]
    return str(row["action"].iloc[0]) if not row.empty else "HOLD"


# ---------------------------------------------------------------------------
# Force close
# ---------------------------------------------------------------------------

def run_force_close(cfg: dict) -> None:
    paper_mode = cfg.get("paper_mode", True)
    positions  = load_open_positions()
    ts         = now_et().strftime("%Y%m%d_%H%M")

    if positions.empty:
        print("[FORCE-CLOSE] No open positions.")
        return

    print(f"\n[FORCE-CLOSE] {now_et().strftime('%H:%M ET')} — closing {len(positions)} position(s)...")

    closes = []
    for _, pos in positions.iterrows():
        ticker = pos["ticker"]
        price  = fetch_last_price(ticker)
        entry  = float(pos["entry_price"]) if pd.notna(pos["entry_price"]) else None
        pnl    = round(price / entry - 1.0, 4) if (price and entry) else None

        print(f"  {ticker}: entry={entry}  last={price}  "
              f"pnl={pnl*100:.2f}%" if pnl is not None else f"  {ticker}: price unavailable")

        closes.append({
            "ticker":       ticker,
            "entry_price":  entry,
            "close_price":  price,
            "qty":          pos.get("qty"),
            "pnl_pct":      pnl,
            "close_time":   ts,
            "reason":       "force_close",
        })

    closes_df   = pd.DataFrame(closes)
    closed_path = LEDGER_DIR / "closed_trades.csv"
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    if closed_path.exists():
        closes_df.to_csv(closed_path, mode="a", header=False, index=False)
    else:
        closes_df.to_csv(closed_path, index=False)
    print(f"  Logged -> {closed_path}")

    save_open_positions(pd.DataFrame(columns=positions.columns))

    if not paper_mode:
        print("  [LIVE] rh_executor not yet wired — close manually on Robinhood.")


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def run_scan(cfg: dict) -> pd.DataFrame:
    paper_mode = cfg.get("paper_mode", True)
    ts         = now_et().strftime("%Y%m%d_%H%M")

    print(f"\n{'='*60}")
    print(f"INTRADAY SCAN — {now_et().strftime('%Y-%m-%d %H:%M')} ET")
    print(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"{'='*60}\n")

    # Gate: PDT
    pdt_tracker = load_pdt_tracker()
    pdt_used    = pdt_count(pdt_tracker, cfg.get("pdt_rolling_calendar_days", 7))
    pdt_max     = cfg.get("pdt_max_trades", 3)
    if pdt_used >= pdt_max:
        print(f"[GATE] PDT exhausted ({pdt_used}/{pdt_max}). No entries.")
        return pd.DataFrame()
    print(f"PDT: {pdt_used}/{pdt_max} trades used")

    # Gate: Macro event
    event_blocked, event_note = macro_event_blackout(cfg.get("macro_blackout_days", 1))
    if event_blocked:
        print(f"[GATE] Macro blackout: {event_note}. No entries.")
        return pd.DataFrame()

    # Gate: VIX
    vix = None
    try:
        vix = fetch_vix()
        print(f"VIX: {vix:.1f}")
        if vix > cfg.get("vix_ceiling", 30):
            print(f"[GATE] VIX {vix:.1f} > {cfg['vix_ceiling']}. No entries.")
            return pd.DataFrame()
    except Exception as e:
        print(f"[VIX] fetch failed ({e}) — VIX gate skipped")

    # Gate: Max positions
    positions      = load_open_positions()
    open_tickers   = set(positions["ticker"].tolist()) if not positions.empty else set()
    slots_left     = cfg.get("max_positions", 2) - len(open_tickers)
    print(f"Positions: {len(open_tickers)}/{cfg.get('max_positions', 2)} open\n")

    # News signals (one call, all tickers)
    news = {}
    if cfg.get("news_enabled", True):
        try:
            from news_watcher import get_news_signals
            news = get_news_signals(cfg["tickers"])
        except Exception as e:
            print(f"[NEWS] {e}")

    print(f"Bulk-fetching {len(cfg['tickers'])} tickers...")
    bar_cache = fetch_1h_bulk(cfg["tickers"], lookback=cfg.get("lookback_bars", 168))
    print(f"  Got data for {len(bar_cache)}/{len(cfg['tickers'])} tickers\n")

    rows = []
    for ticker in cfg["tickers"]:
        print(f"[{ticker}]")
        try:
            df = bar_cache.get(ticker)
            if df is None or df.empty:
                print(f"  no data — skip")
                continue
            s  = compute_score(df, cfg)

            skip_reason = None

            # Gate: daily pipeline
            if cfg.get("daily_gate_enabled", True):
                daily_act = daily_gate_action(ticker)
                if daily_act == "REDUCE":
                    skip_reason = "daily_REDUCE"
                    print(f"  -> SKIP ({skip_reason})")
            else:
                daily_act = "N/A"

            # Gate: news veto
            n = news.get(ticker, {})
            if not skip_reason and n.get("veto"):
                skip_reason = f"news:{n.get('veto_reason', '')}"
                print(f"  -> SKIP ({skip_reason})")

            if skip_reason:
                rows.append({"ticker": ticker, **s, "action": "SKIP",
                             "skip_reason": skip_reason, "daily_action": daily_act,
                             "news_flag": n.get("flag", "-"), "vix": vix})
                continue

            # Score → action
            buy_thr  = cfg.get("buy_threshold", 0.65)
            skip_thr = cfg.get("skip_threshold", 0.40)
            if s["score"] >= buy_thr and ticker not in open_tickers and slots_left > 0:
                action = "BUY"
                slots_left -= 1
            elif s["score"] < skip_thr:
                action = "SKIP"
            elif ticker in open_tickers:
                action = "HOLD"
            else:
                action = "HOLD"

            print(f"  score={s['score']:.3f}  m={s['momentum_signal']:.2f}  "
                  f"v={s['volume_surge']:.2f}  t={s['trend_alignment']:.2f}  "
                  f"vc={s['vol_context']:.2f}  -> {action}")

            last_close = round(float(df["close"].iloc[-1]), 2)
            rows.append({"ticker": ticker, **s, "action": action, "skip_reason": "-",
                         "daily_action": daily_act, "news_flag": n.get("flag", "-"),
                         "vix": vix, "pdt_used": pdt_used, "last_close": last_close})

        except Exception as e:
            print(f"  ERROR: {e}")

    if not rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_df.insert(0, "scan_time", now_et().strftime("%Y-%m-%d %H:%M"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_str  = now_et().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"proposals_intraday_{day_str}.csv"
    if out_path.exists():
        out_df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        out_df.to_csv(out_path, index=False)
    print(f"\nAppended {len(out_df)} rows -> {out_path}")

    # Paper ledger: record BUY entries
    if paper_mode:
        buys = out_df[out_df["action"] == "BUY"]
        if not buys.empty:
            LEDGER_DIR.mkdir(parents=True, exist_ok=True)
            pos_path = LEDGER_DIR / "positions.csv"
            new_rows = []
            for _, row in buys.iterrows():
                price  = row.get("last_close")
                stop   = round(price * (1 - cfg.get("stop_loss_pct",    0.01)),  2) if price else None
                target = round(price * (1 + cfg.get("profit_target_pct", 0.015)), 2) if price else None
                new_rows.append({
                    "ticker":      row["ticker"],
                    "entry_price": price,
                    "qty":         None,
                    "entry_time":  ts,
                    "stop":        stop,
                    "target":      target,
                })
            new_df = pd.DataFrame(new_rows)
            if pos_path.exists():
                new_df.to_csv(pos_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(pos_path, index=False)
            print(f"  Recorded {len(buys)} BUY(s) -> {pos_path}")
    else:
        print("  [LIVE] rh_executor not yet wired.")

    return out_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-close", action="store_true", help="Close all open positions")
    parser.add_argument("--scan",        action="store_true", help="Force scan mode")
    args = parser.parse_args()

    cfg = load_config()
    t   = time_et_str()

    if args.force_close or (not args.scan and t >= cfg.get("force_close_at", "15:25")):
        run_force_close(cfg)
    else:
        run_scan(cfg)


if __name__ == "__main__":
    main()
