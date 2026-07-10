# investment-pipeline — Project Context

## What This Is
Local-first quant scoring system for a Fidelity Rollover IRA ($126k).
Python 3.14, PyTorch CPU, zero cloud dependencies, zero paid data subscriptions.
Two pipelines: daily (Fidelity IRA) + intraday (Robinhood agentic paper mode).

---

## Accounts

| Account | Broker | Value | Mode |
|---|---|---|---|
| Rollover IRA ••••4945 | Fidelity | ~$126k | Manual — Claude proposes, Frank approves |
| Agentic ••••6090 | Robinhood | $0 (funding ~$2k soon) | Paper now → live after 2 weeks signals |

Current Fidelity positions: FDRXX (97.79% cash), CACI (2 shares), META (3 shares).

---

## Daily Pipeline

**Run:** Windows Task Scheduler → "Investment Pipeline Scheduler" → 4:30pm ET daily
**Script:** `pipeline.py`
**Config:** `config.json` (interval=1d, pred_len=10, kronos-mini, 16 tickers)
**Output:** `output/proposals_YYYYMMDD_HHMMSS.csv`, `output/factor_log.csv`

6 factors (equal weight): forecast_edge, path_consistency, vol_context, trend_alignment, lt_quality, contract_signal
Thresholds: BUY ≥ 0.70 | HOLD 0.45–0.70 | REDUCE < 0.45

Kronos IC validated: **1d/10-candle = +0.145** (usable). 1h/12-candle = -0.130 (discarded).

---

## Intraday Pipeline

**Run:** Windows Task Scheduler → "Intraday Pipeline" → 9:45, 11:00, 12:30, 14:00, 15:25 ET Mon–Fri
**Script:** `intraday_pipeline.py`
**Config:** `intraday_config.json` (max_positions=2, buy_thr=0.65, paper_mode=true)
**Output:** `output/intraday/proposals_intraday_YYYYMMDD.csv` (daily append, scan_time column)
**Ledger:** `ledger/intraday/positions.csv`, `closed_trades.csv`, `pdt_tracker.json`

4 factors: momentum_signal, volume_surge, trend_alignment, vol_context
**No Kronos** — 1h IC = -0.130, anti-predictive. Daily Kronos proposals used as gate only.

Gates (order): PDT budget → macro blackout → VIX >30 → daily gate (REDUCE=skip) → news veto → max positions

Force-close: script auto-detects time ≥15:25 ET and closes all positions.
`rh_executor.py` — NOT YET BUILT. Build after 2 weeks of paper signals validate edge.

---

## Key Files

| File | Purpose |
|---|---|
| `pipeline.py` | Daily scorer (6 factors + optimizer) |
| `intraday_pipeline.py` | Intraday scorer (4 factors, paper mode) |
| `intraday_config.json` | Intraday config |
| `config.json` | Daily config |
| `kronos_forecast.py` | Kronos inference wrapper |
| `robinhood_fetcher.py` | RH data fetch utility (standalone, not wired to pipeline) |
| `backtest.py` | Walk-forward IC backtest (`--mode kronos --interval 1h --horizon 12`) |
| `ledger.py` | Daily paper ledger |
| `news_watcher.py` | FinBERT news veto (7 categories) |
| `macro_calendar.json` | FOMC/CPI/NFP dates — verify yearly |
| `run_scheduler.py` | Called by Task Scheduler for daily pipeline |

---

## Python Environment

Always use: `C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe`
Shell default (Hermes venv, Python 3.11) does NOT have torch — will fail silently.
torch 2.12.1+cpu lives in Python 3.14 site-packages only.

---

## Validated Decisions

- Hold threshold: 0.45 (not 0.40) — Sharpe 0.55→0.71, see `notes/hermes_notes/2026-06-15-CARRY-FORWARD.md`
- FRED fetch: 30s timeout, 3 retries, stale-cache fallback (hardened 2026-06-15)
- Kronos 1h discarded permanently — IC -0.130 confirmed twice (Jun + Jul 2026)
- PyPortfolioOpt Efficient Frontier + Ledoit-Wolf shrinkage for daily sizing
- Intraday: flat sizing (position_size_pct=0.40), no optimizer needed at 2 positions

---

## What NOT to Touch

- Daily pipeline config/thresholds — validated, don't change without re-running backtest
- `factor_log.csv` — append-only training data, never delete
- `macro_calendar.json` — verify dates yearly (BLS/Fed calendar)

---

---

## Changelog

| Date | File | Change | Reason | Commit |
|---|---|---|---|---|
| 2026-07-06 | `daily_brief.py` | Added `fallback_brief()` — data-only brief when Ollama unreachable | Brief was stale Jun 9→Jul 6; Ollama down, no fallback | f243b3e |
| 2026-07-06 | `run_scheduler.py` | Added `ensure_ollama()` to auto-start Ollama before brief | Ollama needed but not running on schedule | f243b3e |
| 2026-07-10 | `intraday_pipeline.py` | Audited: found qty sizing gap (hardcoded None, no position sizing logic) | closed_trades.csv missing qty/PnL; rh_executor.py still TODO | — |

**Known issues:**
- Intraday position sizing: qty hardcoded to None. No $ allocation logic. Needs flat sizing or risk framework before live.
- SPY/QQQ not in daily proposals (tracked but not scored — benchmark only?).
- BAH forecast anomaly (+33.94% on Jul10 scan) — likely data artifact.

---

## Pending

- [ ] Fund Robinhood agentic ••••6090 (~$2k, Frank's task)
- [ ] Add position sizing to intraday BUY entries (flat $5k/trade or risk-based)
- [ ] Add 100+ tickers to `intraday_config.json` tickers array
- [ ] Run IC check on intraday proposals after 2 weeks
- [ ] Build `rh_executor.py` after paper signals validated
- [ ] Run `/wrapup` and push to Brain at session end
