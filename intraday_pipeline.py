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
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

CONFIG_PATH = Path("intraday_config.json")
OUTPUT_DIR  = Path("output/intraday")
LEDGER_DIR  = Path("ledger/intraday")

ET = ZoneInfo("America/New_York")

BARS_PER_DAY_1H = 6.5  # US session hours
ANNUALIZE_1H    = np.sqrt(252 * BARS_PER_DAY_1H)  # ~40.5


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    # Sanity: total allocation must fit inside equity
    # (max_positions * position_size_pct = 0.80 <= 1.0 with current config)
    total_alloc = cfg.get("max_positions", 2) * cfg.get("position_size_pct", 0.40)
    if total_alloc > 1.0:
        raise SystemExit(f"[CONFIG] max_positions * position_size_pct = "
                         f"{total_alloc:.2f} > 1.0 — would over-allocate equity")
    return cfg


def _size_position(price, size_pct: float, equity: float):
    """Whole-share qty for a flat size_pct slice of equity. None on bad price;
    0 when one share exceeds the allocation (caller logs + skips)."""
    if not price or price <= 0:
        return None
    return int((equity * size_pct) // price)


def now_et() -> datetime:
    return datetime.now(ET)


def time_et_str() -> str:
    return now_et().strftime("%H:%M")


def _notify(title: str, message: str) -> None:
    """Best-effort Windows toast notification; never raises. Task Scheduler
    runs headless, so this is the only way a failure surfaces to Frank."""
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=10)
    except Exception:
        pass


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
                # Staleness: during a live weekday session a >2h-old last bar
                # means the feed is broken for this ticker — don't score it.
                now = now_et()
                in_session = (now.weekday() < 5
                              and "09:30" <= now.strftime("%H:%M") <= "16:00")
                last_ts = pd.Timestamp(df["timestamps"].iloc[-1])
                last_ts = (last_ts.tz_localize(ET) if last_ts.tzinfo is None
                           else last_ts.tz_convert(ET))
                age = (pd.Timestamp(now) - last_ts).total_seconds() / 3600.0
                if in_session and age > 2:
                    print(f"  [STALE] {ticker} last bar {age:.1f}h old — skipped")
                    continue
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
    # Exclude the still-forming bar: comparing a partial hour's volume to full
    # 1h baselines deflates the score. If the last bar's hour hasn't elapsed,
    # score the previous (completed) bar with the baseline shifted back one.
    partial = False
    if "timestamps" in df.columns:
        last_ts = pd.Timestamp(df["timestamps"].iloc[-1])
        now = pd.Timestamp(now_et())
        now = now.tz_localize(None) if last_ts.tzinfo is None else now.tz_convert(ET)
        last_ts = last_ts if last_ts.tzinfo is None else last_ts.tz_convert(ET)
        elapsed_h = (now - last_ts).total_seconds() / 3600.0
        partial = 0 <= elapsed_h < 1.0
    if partial and len(df) >= window + 2:
        avg = float(df["volume"].iloc[-window - 2:-2].mean())
        cur = float(df["volume"].iloc[-2])
    else:
        avg = float(df["volume"].iloc[-window - 1:-1].mean())
        cur = float(df["volume"].iloc[-1])
    if avg <= 0:
        return 0.5
    ratio = cur / avg
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
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print("[PDT] pdt_tracker.json corrupt, resetting to empty tracker "
                  "— history since last valid save lost")
    return {"trades": []}


def pdt_count(tracker: dict, rolling_days: int = 7) -> int:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=rolling_days)
    return sum(1 for t in tracker["trades"] if pd.Timestamp(t) >= cutoff)


def save_pdt_tracker(tracker: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_DIR / "pdt_tracker.json", "w") as f:
        json.dump(tracker, f, indent=2)


def record_pdt_trades(tracker: dict, n_buys: int) -> None:
    """Count each entry as one PDT-relevant trade (conservative). Persist."""
    if n_buys <= 0:
        return
    ts = now_et().isoformat()
    tracker["trades"].extend([ts] * n_buys)
    save_pdt_tracker(tracker)


def load_open_positions() -> pd.DataFrame:
    path = LEDGER_DIR / "positions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["ticker", "entry_price", "qty", "entry_time", "stop", "target"])


def save_open_positions(df: pd.DataFrame) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEDGER_DIR / "positions.csv", index=False)


def daily_gate_action(ticker: str) -> str:
    """Read latest daily proposals CSV; 'N/A' when the ticker isn't covered
    by the daily universe (was silently 'HOLD', hiding the coverage gap —
    only REDUCE actually gates entries, so this is observability, not a
    behavior change for covered tickers)."""
    candidates = sorted(p for p in Path("output").glob("proposals_*.csv")
                        if "intraday" not in p.name)
    if not candidates:
        return "N/A"
    df = pd.read_csv(candidates[-1])
    row = df[df["ticker"] == ticker]
    return str(row["action"].iloc[0]) if not row.empty else "N/A"


# ---------------------------------------------------------------------------
# Force close
# ---------------------------------------------------------------------------

def run_force_close(cfg: dict) -> None:
    paper_mode = cfg.get("paper_mode", True)
    positions  = load_open_positions()
    ts         = now_et().strftime("%Y%m%d_%H%M")

    if positions.empty:
        # Idempotency guard: positions.csv is cleared after a close, so a
        # second --force-close run no-ops here instead of double-logging trades.
        print("[FORCE-CLOSE] No open positions.")
        return

    print(f"\n[FORCE-CLOSE] {now_et().strftime('%H:%M ET')} — closing {len(positions)} position(s)...")

    closes = []
    retained = []
    failed_tickers = []
    for _, pos in positions.iterrows():
        ticker = pos["ticker"]
        price  = fetch_last_price(ticker)
        if price is None:
            # Fallback: latest 1h bar close via the existing bulk fetcher
            bars   = fetch_1h_bulk([ticker], lookback=1)
            bar_df = bars.get(ticker)
            if bar_df is not None and not bar_df.empty:
                price = float(bar_df["close"].iloc[-1])
                print(f"  [FORCE-CLOSE] {ticker}: fetch_last_price failed, "
                      f"used latest 1h bar close ${price:.2f}")

        if price is None:
            # Both sources failed — KEEP the position (never wipe an
            # unpriced close: PnL would be permanently unrecoverable),
            # retry on the next scheduled run.
            print(f"  [FORCE-CLOSE-FAILED] {ticker}: no price available "
                  f"(fetch_last_price and 1h bar both failed) — "
                  f"position RETAINED, will retry next run")
            retained.append(pos)
            failed_tickers.append(ticker)
            continue

        entry  = float(pos["entry_price"]) if pd.notna(pos["entry_price"]) else None
        pnl    = round(price / entry - 1.0, 4) if (price and entry) else None
        qty    = pos.get("qty")
        qty    = int(qty) if pd.notna(qty) else None
        pnl_dollars = (round((price - entry) * qty, 2)
                       if (pnl is not None and qty is not None) else None)

        print(f"  {ticker}: entry={entry}  last={price}  "
              f"pnl={pnl*100:.2f}%" if pnl is not None else f"  {ticker}: price unavailable")

        closes.append({
            "ticker":       ticker,
            "entry_price":  entry,
            "close_price":  price,
            "qty":          qty,
            "pnl_pct":      pnl,
            "pnl_dollars":  pnl_dollars,
            "close_time":   ts,
            "reason":       "force_close",
        })

    # Only positions with a real close price are logged to closed_trades
    if closes:
        closes_df   = pd.DataFrame(closes)
        closed_path = LEDGER_DIR / "closed_trades.csv"
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        if closed_path.exists():
            closes_df.to_csv(closed_path, mode="a", header=False, index=False)
        else:
            closes_df.to_csv(closed_path, index=False)
        print(f"  Logged -> {closed_path}")

    # Only successfully-priced positions are cleared; retained (unpriced)
    # ones stay in positions.csv so the next run retries their close.
    remaining = pd.DataFrame(retained, columns=positions.columns) if retained \
        else pd.DataFrame(columns=positions.columns)
    save_open_positions(remaining)

    if not paper_mode:
        print("  [LIVE] rh_executor not yet wired — close manually on Robinhood.")

    if failed_tickers:
        _notify("Intraday force-close FAILED",
                f"{len(failed_tickers)} position(s) could not be priced: "
                f"{', '.join(failed_tickers)} — retained for retry")
        sys.exit(1)


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

    # Gate: entry time (scan still scores/logs for visibility)
    no_entry_before = cfg.get("no_entry_before", "09:45")
    entries_allowed = time_et_str() >= no_entry_before and now_et().weekday() < 5
    if not entries_allowed:
        print(f"[GATE] before {no_entry_before} ET or weekend — "
              f"entries disabled, scoring only")

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

    # Gate observability: how much of the intraday universe the daily gate
    # actually covers (uncovered tickers can never be REDUCE-skipped by it)
    if cfg.get("daily_gate_enabled", True):
        n_covered = sum(1 for t in cfg["tickers"] if daily_gate_action(t) != "N/A")
        print(f"[GATE] daily gate covers {n_covered}/{len(cfg['tickers'])} tickers")

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
    scan_errors = {}
    for ticker in cfg["tickers"]:
        print(f"[{ticker}]")
        try:
            df = bar_cache.get(ticker)
            if df is None or df.empty:
                print(f"  no data — skip")
                scan_errors[ticker] = "no data (fetch failed or empty)"
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
                             "news_flag": n.get("flag", "-"), "news_sent": n.get("sent", 0.0),
                             "vix": vix})
                continue

            # Score → action
            buy_thr  = cfg.get("buy_threshold", 0.65)
            skip_thr = cfg.get("skip_threshold", 0.40)
            if (s["score"] >= buy_thr and ticker not in open_tickers
                    and slots_left > 0 and entries_allowed):
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
                         "news_sent": n.get("sent", 0.0),
                         "vix": vix, "pdt_used": pdt_used, "last_close": last_close})

        except Exception as e:
            print(f"  ERROR: {e}")
            scan_errors[ticker] = str(e)

    if not rows:
        if scan_errors:
            # Reached the ticker loop (passed all global gates) but every
            # ticker failed — e.g. yfinance throttled the whole bulk fetch.
            # A legitimately all-gated/all-skipped scan never lands here:
            # SKIP rows (daily-gate/news veto) still append to `rows` above,
            # and a global gate (PDT/macro/VIX) returns before this point.
            print(f"\n[SCAN-FAILURE] 0/{len(cfg['tickers'])} tickers scored — all failed:")
            for t, reason in scan_errors.items():
                print(f"  {t}: {reason}")
            _notify("Intraday scan FAILED",
                    f"0/{len(cfg['tickers'])} tickers scored — check logs")
            sys.exit(1)
        return pd.DataFrame()

    out_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    scan_time = now_et().strftime("%Y-%m-%d %H:%M")
    out_df.insert(0, "scan_time", scan_time)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_str  = now_et().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"proposals_intraday_{day_str}.csv"
    if out_path.exists():
        # Idempotency: a re-run within the same minute must not double-log
        existing_times = set(pd.read_csv(out_path)["scan_time"].astype(str))
        if scan_time in existing_times:
            print(f"\nduplicate scan_time {scan_time} — skipping append")
        else:
            out_df.to_csv(out_path, mode="a", header=False, index=False)
            print(f"\nAppended {len(out_df)} rows -> {out_path}")
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
                qty    = _size_position(price, cfg.get("position_size_pct", 0.40),
                                        cfg.get("paper_equity", 2000))
                if qty == 0:
                    print(f"  [SIZE] {row['ticker']} skipped: 1 share (${price}) "
                          f"exceeds allocation")
                    continue  # no phantom position; proposals row keeps the signal
                stop   = round(price * (1 - cfg.get("stop_loss_pct",    0.01)),  2) if price else None
                target = round(price * (1 + cfg.get("profit_target_pct", 0.015)), 2) if price else None
                new_rows.append({
                    "ticker":      row["ticker"],
                    "entry_price": price,
                    "qty":         qty,
                    "entry_time":  ts,
                    "stop":        stop,
                    "target":      target,
                })
            # Idempotency: drop rows whose (ticker, entry_time) already exists
            if new_rows and pos_path.exists():
                existing_pos = pd.read_csv(pos_path)
                seen = set(zip(existing_pos["ticker"].astype(str),
                               existing_pos["entry_time"].astype(str)))
                new_rows = [r for r in new_rows
                            if (str(r["ticker"]), str(r["entry_time"])) not in seen]
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                if pos_path.exists():
                    new_df.to_csv(pos_path, mode="a", header=False, index=False)
                else:
                    new_df.to_csv(pos_path, index=False)
                print(f"  Recorded {len(new_rows)} BUY(s) -> {pos_path}")
                # PDT gate was inert without this: entries must count against the budget
                record_pdt_trades(pdt_tracker, len(new_rows))
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

    try:
        if args.force_close or (not args.scan and t >= cfg.get("force_close_at", "15:25")):
            run_force_close(cfg)
        else:
            run_scan(cfg)
    except Exception:
        # Task Scheduler runs headless — surface the crash instead of dying silently.
        # SystemExit (from run_scan's SCAN-FAILURE or run_force_close's
        # FORCE-CLOSE-FAILED paths) is a BaseException, not an Exception, so
        # it passes through here untouched — those paths already logged and
        # notified their own specific failure.
        traceback.print_exc()
        _notify("Intraday pipeline CRASHED", "Intraday pipeline CRASHED — check logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
