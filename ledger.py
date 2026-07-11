"""
ledger.py
Paper-trade ledger: records pipeline proposals, simulates fills at the next
session open, marks positions to market, and tracks an equity curve.

By June 17 (rollover arrival) this gives real evidence of whether the
pipeline's proposals make money — not vibes.

Files (in ledger/):
    proposals_log.csv  every proposal the pipeline emitted (with filled flag)
    positions.csv      open paper positions
    closed_trades.csv  realized round trips with P&L
    equity_curve.csv   daily cash + market value
    state.json         cash balance

Fill rules (simple, conservative):
    BUY     -> filled at next trading day's open, sized to target_value,
               skipped if already holding the ticker (no pyramiding)
    REDUCE  -> closes the position at next trading day's open
    HOLD    -> no action

Usage:
    python ledger.py mark      # process pending fills + mark to market
    python ledger.py status    # print positions and P&L summary
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

LEDGER_DIR = Path("ledger")
PROPOSALS = LEDGER_DIR / "proposals_log.csv"
POSITIONS = LEDGER_DIR / "positions.csv"
CLOSED = LEDGER_DIR / "closed_trades.csv"
EQUITY = LEDGER_DIR / "equity_curve.csv"
STATE = LEDGER_DIR / "state.json"

# 2026 NYSE full-day closures. macro_calendar.json carries no holiday list,
# so these are hardcoded — verify yearly against nyse.com/markets/hours-calendars.
NYSE_HOLIDAYS_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Washington's Birthday
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
]


def _load_state(starting_cash: float) -> dict:
    if STATE.exists():
        with open(STATE) as f:
            return json.load(f)
    return {"cash": starting_cash}


def _save_state(state: dict):
    LEDGER_DIR.mkdir(exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)


def _load_csv(path: Path, columns: list) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def record(proposals: pd.DataFrame):
    """Append today's pipeline proposals (call after each pipeline run)."""
    if proposals.empty:
        return
    LEDGER_DIR.mkdir(exist_ok=True)
    rows = proposals[["ticker", "action", "adj_score", "target_value"]].copy()
    rows.insert(0, "proposal_date", datetime.now().strftime("%Y-%m-%d"))
    rows["filled"] = False
    log = _load_csv(PROPOSALS, list(rows.columns))
    # one proposal per ticker per day — latest run wins
    log = log[~((log["proposal_date"] == rows["proposal_date"].iloc[0])
                & (log["ticker"].isin(rows["ticker"])))]
    log = pd.concat([log, rows], ignore_index=True)
    log.to_csv(PROPOSALS, index=False)
    print(f"[LEDGER] Recorded {len(rows)} proposals -> {PROPOSALS}")


def _today_prices(tickers: list) -> tuple[dict, dict]:
    """Returns ({ticker: (bar_date, open_px)}, {ticker: (bar_date, close_px)}).
    bar_date (YYYY-MM-DD) is the trading date the price belongs to, so callers
    can refuse to fill a proposal unless a genuinely NEW session has occurred."""
    if not tickers:
        return {}, {}
    data = yf.download(tickers, period="5d", interval="1d",
                       progress=False, auto_adjust=True, group_by="ticker")
    opens, closes = {}, {}
    for t in tickers:
        try:
            sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            o = sub["Open"].dropna()
            c = sub["Close"].dropna()
            opens[t] = (o.index[-1].strftime("%Y-%m-%d"), float(o.iloc[-1]))
            closes[t] = (c.index[-1].strftime("%Y-%m-%d"), float(c.iloc[-1]))
        except Exception as e:
            print(f"[LEDGER] no price for {t}: {e}")
    return opens, closes


def _flag_stale(log: pd.DataFrame, today: str, max_gap_days: int = 1) -> pd.DataFrame:
    """Proposals older than max_gap_days trading-ish days are expired, not filled.
    A missed scheduler run must not fill a days-old signal at today's open."""
    log = log.copy()
    gap = (pd.Timestamp(today) - pd.to_datetime(log["proposal_date"])).dt.days
    log["expired"] = (~log["filled"].astype(bool)) & (gap > max_gap_days + 2)  # +2 tolerates weekends
    return log


def mark(starting_cash: float = 124_000, slippage_bps: float = 10.0):
    """
    Process pending fills from prior proposals, then mark to market.
    Fills pay slippage: buys at open*(1+bps), sells at open*(1-bps) —
    exact-open fills are fantasy fills and flatter the strategy.
    """
    now = datetime.now()
    if now.weekday() >= 5:
        print("[LEDGER] Skipping mark(): weekend, no trading session.")
        return None
    today = now.strftime("%Y-%m-%d")
    if today in NYSE_HOLIDAYS_2026:
        print(f"[LEDGER] Skipping mark(): NYSE holiday {today}, no trading session.")
        return None
    slip = slippage_bps / 10_000
    state = _load_state(starting_cash)
    log = _load_csv(PROPOSALS, ["proposal_date", "ticker", "action",
                                "adj_score", "target_value", "filled"])
    pos = _load_csv(POSITIONS, ["ticker", "entry_date", "entry_price",
                                "shares", "cost_basis"])
    closed = _load_csv(CLOSED, ["ticker", "entry_date", "entry_price",
                                "exit_date", "exit_price", "shares",
                                "pnl", "pnl_pct"])

    # Expire stale proposals: a missed scheduler run must not fill a days-old
    # signal at today's open as if it were fresh.
    log = _flag_stale(log, today)
    for idx in log.index[log["expired"].astype(bool)]:
        print(f"[LEDGER] EXPIRED (not filled): {log.loc[idx, 'ticker']} "
              f"proposal from {log.loc[idx, 'proposal_date']}")
        log.loc[idx, "filled"] = True  # processed (as expired), never enters a position

    pending = log[(~log["filled"].astype(bool)) & (log["proposal_date"] < today)]
    tickers_needed = sorted(set(pending["ticker"]) | set(pos["ticker"]))
    opens, closes = _today_prices(tickers_needed)

    held = set(pos["ticker"])
    for idx, row in pending.iterrows():
        t, act = row["ticker"], row["action"]
        o = opens.get(t)
        if o is None:
            continue
        bar_date, px = o
        if bar_date <= str(row["proposal_date"]):
            # no NEW session since the proposal was logged — leave unfilled for retry
            continue
        px = px * (1 + slip) if act == "BUY" else px * (1 - slip)
        executed = act != "BUY" or t in held
        if act == "BUY" and t not in held:
            shares = int(row["target_value"] // px) if row["target_value"] > 0 else 0
            cost = shares * px
            executed = shares > 0 and cost <= state["cash"]
            if executed:
                state["cash"] -= cost
                pos = pd.concat([pos, pd.DataFrame([{
                    "ticker": t, "entry_date": bar_date, "entry_price": round(px, 2),
                    "shares": shares, "cost_basis": round(cost, 2),
                }])], ignore_index=True)
                held.add(t)
                print(f"[LEDGER] FILL BUY {t}: {shares} @ {px:.2f}")
            else:
                print(f"[LEDGER] BUY {t} SKIPPED: shares={shares} "
                      f"cost={cost:.2f} cash={state['cash']:.2f}")
        elif act == "REDUCE" and t in held:
            p = pos[pos["ticker"] == t].iloc[0]
            proceeds = p["shares"] * px
            pnl = proceeds - p["cost_basis"]
            state["cash"] += proceeds
            closed = pd.concat([closed, pd.DataFrame([{
                "ticker": t, "entry_date": p["entry_date"],
                "entry_price": p["entry_price"], "exit_date": bar_date,
                "exit_price": round(px, 2), "shares": p["shares"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / p["cost_basis"] * 100, 2),
            }])], ignore_index=True)
            pos = pos[pos["ticker"] != t]
            held.discard(t)
            print(f"[LEDGER] FILL SELL {t}: {p['shares']} @ {px:.2f}  pnl={pnl:+.2f}")
        log.loc[idx, "filled"] = executed

    # mark to market
    mv = 0.0
    for _, p in pos.iterrows():
        c = closes.get(p["ticker"])
        mv += p["shares"] * (c[1] if c is not None else p["entry_price"])
    equity = state["cash"] + mv

    # equity-curve row stamped with the BAR's trading date, never wall-clock now()
    mark_date = max((d for d, _ in closes.values()), default=today)
    eq = _load_csv(EQUITY, ["date", "cash", "market_value", "equity"])
    eq = eq[eq["date"] != mark_date]
    eq = pd.concat([eq, pd.DataFrame([{
        "date": mark_date, "cash": round(state["cash"], 2),
        "market_value": round(mv, 2), "equity": round(equity, 2),
    }])], ignore_index=True)

    LEDGER_DIR.mkdir(exist_ok=True)
    log.to_csv(PROPOSALS, index=False)
    pos.to_csv(POSITIONS, index=False)
    closed.to_csv(CLOSED, index=False)
    eq.to_csv(EQUITY, index=False)
    _save_state(state)

    print(f"[LEDGER] {today}  cash=${state['cash']:,.0f}  positions=${mv:,.0f}  "
          f"equity=${equity:,.0f}  ({(equity / starting_cash - 1) * 100:+.2f}% since start)")
    return equity


def status():
    pos = _load_csv(POSITIONS, ["ticker", "entry_date", "entry_price", "shares", "cost_basis"])
    closed = _load_csv(CLOSED, ["ticker", "entry_date", "entry_price", "exit_date",
                                "exit_price", "shares", "pnl", "pnl_pct"])
    eq = _load_csv(EQUITY, ["date", "cash", "market_value", "equity"])

    print("\n=== OPEN POSITIONS ===")
    print(pos.to_string(index=False) if not pos.empty else "(none)")
    print("\n=== CLOSED TRADES ===")
    print(closed.to_string(index=False) if not closed.empty else "(none)")
    if not closed.empty:
        print(f"\nRealized P&L: ${closed['pnl'].sum():+,.2f}  "
              f"win rate: {(closed['pnl'] > 0).mean() * 100:.0f}%")
    print("\n=== EQUITY CURVE (last 10) ===")
    print(eq.tail(10).to_string(index=False) if not eq.empty else "(none)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-trade ledger")
    parser.add_argument("cmd", choices=["mark", "status"])
    parser.add_argument("--cash", type=float, default=124_000, help="Starting cash")
    args = parser.parse_args()
    if args.cmd == "mark":
        mark(starting_cash=args.cash)
    else:
        status()
