# Investment Pipeline

Local-first quant scoring system for a Fidelity Rollover IRA  
Python 3.14, PyTorch CPU, zero cloud dependencies, zero paid data subscriptions.  
Two pipelines: **daily** (Fidelity IRA) + **intraday** (Robinhood agentic paper mode).

## Purpose
Generate daily/hourly trading signals using technical factors + macro gates. Paper-trade to validate edge before live execution.

## Edge Thesis
Momentum (7-bar returns) + trend alignment (EMA ratio) + volatility context (RV20) identify mean-reversion setups with positive expected return. Kronos (daily, 1d/10-candle) adds price-forecast layer. Intraday gates (PDT, VIX, macro, news) reduce false positives. Target Sharpe ≥0.70.

## Architecture

```
yfinance (1h/1d bars) → compute_score() [4-6 factors] → apply gates [PDT/macro/VIX/daily/news]
  → BUY/HOLD/REDUCE/SKIP action → paper ledger (positions.csv + equity_curve.csv)
  → daily_brief.py (Ollama LLM or data fallback) → output/proposals_*.csv
```

**Pipeline layers:**
1. Factor computation — momentum, volume, trend, volatility
2. Kronos (daily only) — transformer-based price forecast (10-candle horizon)
3. Gating — PDT budget, macro calendar, VIX, daily regime, news sentiment
4. Position management — flat sizing (intraday) / optimized (daily)
5. Reporting — CSV proposals, brief, equity curve, trade ledger

## Quick Start

### Manual Runs
```bash
# Python 3.14 ONLY — torch 2.12.1+cpu lives here
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe pipeline.py           # Daily scan
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe intraday_pipeline.py  # Hourly scan
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe intraday_pipeline.py --force-close  # End-of-day close
```

### Automated (Windows Task Scheduler)
- **Daily:** "Investment Pipeline Scheduler" → 4:30pm ET Mon–Fri → runs `run_scheduler.py --now`
- **Intraday:** "Intraday Pipeline" → 9:45, 11:00, 12:30, 14:00, 15:25 ET Mon–Fri → runs `intraday_pipeline.py`

### Check Outputs
- Daily proposals: `output/proposals_YYYYMMDD_HHMMSS.csv`
- Intraday proposals: `output/intraday/proposals_intraday_YYYYMMDD.csv`
- Daily brief: `output/briefs/brief_YYYYMMDD.md`
- Equity curves: `ledger/equity_curve.csv`, `ledger/intraday/positions.csv`

## Installation

```bash
# Use Python 3.14
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe -m pip install -r requirements.txt

# Ollama (optional, for daily brief)
# Download from ollama.ai, then:
ollama pull llama3.1:8b
```

## Accounts

| Account | Broker | Value | Mode |
|---------|--------|-------|------|
| Rollover IRA •••• | Fidelity || Manual — Claude proposes, Frank approves |
| Agentic •••• | Robinhood | $0 (funding ~$2k soon) | Paper now → live after 2 weeks signals |

Current Fidelity positions: FDRXX (97.79% cash), CACI (2 shares), META (3 shares).

## Pipelines

### Daily Pipeline
- **Script:** `pipeline.py`
- **Config:** `config.json` (interval=1d, pred_len=10, kronos-mini, 16 tickers)
- **Output:** `output/proposals_YYYYMMDD_HHMMSS.csv`, `output/factor_log.csv`
- **6 factors (equal weight):** forecast_edge, path_consistency, vol_context, trend_alignment, lt_quality, contract_signal
- **Thresholds:** BUY ≥ 0.70 | HOLD 0.45–0.70 | REDUCE < 0.45
- **Kronos IC — superseded, do not cite the old +0.145 figure.** Re-measured during the 2026-07-11 audit fix: honest (non-overlapping-window, median-real-path) IC is **+0.061**, hit rate 50% (coin flip). See `CLAUDE.md` Daily Pipeline section.

### Intraday Pipeline
- **Script:** `intraday_pipeline.py`
- **Config:** `intraday_config.json` (max_positions=2, buy_thr=0.65, paper_mode=true)
- **Output:** `output/intraday/proposals_intraday_YYYYMMDD.csv` (daily append, scan_time column)
- **Ledger:** `ledger/intraday/positions.csv`, `closed_trades.csv`, `pdt_tracker.json`
- **4 factors:** momentum_signal, volume_surge, trend_alignment, vol_context
- **No Kronos** — 1h IC = -0.130, anti-predictive. Daily Kronos proposals used as gate only.
- **Gates (order):** PDT budget → macro blackout → VIX >30 → daily gate (REDUCE=skip) → news veto → max positions
- **Force-close:** Auto-detects time ≥15:25 ET and closes all positions.
- Position sizing (flat `paper_equity x position_size_pct`, whole shares) and dollar P&L in `closed_trades.csv` fixed 2026-07-11 (audit-fix Phase B). See `CLAUDE.md` changelog.

## Key Files

| Category | Files |
|----------|-------|
| **Core Pipeline** | `pipeline.py`, `intraday_pipeline.py`, `run_scheduler.py`, `kronos_forecast.py`, `ledger.py`, `news_watcher.py`, `daily_brief.py` |
| **Config** | `config.json`, `intraday_config.json`, `macro_calendar.json` |
| **Research** | `backtest.py`, `ic_report.py`, `learn_weights.py` |
| **Signal Utils** | `edgar_watcher.py`, `quiver_congress_watchlist.py`, `usaspending_watcher.py` |
| **Dead/Archive** | `debug.py`, `make_brief.py`, `make_brief_fixed.py`, `research_script.py`, `research_script_final.py`, `robinhood_fetcher.py`, `build_basket.py` |
| **Model Library** | `Kronos/` (model weights, finetune scripts, tests, webui, examples) |

## Python Environment
**Always use:** `C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe`  
Shell default (Hermes venv, Python 3.11) does NOT have torch — will fail silently.

## Validated Decisions
- Hold threshold: 0.45 (not 0.40) — Sharpe 0.55→0.71, see `notes/hermes_notes/2026-06-15-CARRY-FORWARD.md`
- FRED fetch: 30s timeout, 3 retries, stale-cache fallback (hardened 2026-06-15)
- Kronos 1h discarded permanently — IC -0.130 confirmed twice (Jun + Jul 2026)
- PyPortfolioOpt Efficient Frontier + Ledoit-Wolf shrinkage for daily sizing
- Intraday: flat sizing (position_size_pct=0.40), no optimizer needed at 2 positions

## Tests

```bash
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe -m pytest test_ic_report.py test_intraday.py test_learn_weights.py test_ledger.py test_pipeline.py test_scoring.py -v
```

**48 tests** across 6 files (scoped run — bare `pytest` also collects the
vendored `Kronos/` submodule's own test suite, which needs a different
setup and isn't part of this project's coverage):

| File | Covers |
|---|---|
| `test_pipeline.py` | Performance stats, rank IC, BUY/HOLD/REDUCE classifier + regime gating, forecast-horizon math, saturating Kronos annualization, cross-sectional factor z-scoring |
| `test_scoring.py` | NaN-safe vol_context (daily + intraday), trend-gate/scorer band agreement, saturating mu, configurable VIX ceiling, daily-gate coverage (`N/A` for uncovered tickers) |
| `test_ledger.py` | Weekend/stale-fill rejection, honest `filled` flag |
| `test_intraday.py` | PDT tracker persistence + corrupt-state recovery, position sizing, partial-bar volume scoring, RH timestamp tz conversion |
| `test_ic_report.py` | Next-session entry-fill anchoring for forward-return calculation |
| `test_learn_weights.py` | Negative-IC factors get zero weight (never invert signal) |

All synthetic inputs, hand-derived expected values — no network, no live
data, no model inference.

## What NOT to Touch
- Daily pipeline config/thresholds — validated, don't change without re-running backtest
- `factor_log.csv` — append-only training data, never delete
- `macro_calendar.json` — verify dates yearly (BLS/Fed calendar)

## Changelog
See `CLAUDE.md` for full changelog with dates, files, reasons, and commits.

## Pending
- [ ] Fund Robinhood agentic •••• (~$2k)
- [ ] Add position sizing to intraday BUY entries
- [ ] Add 100+ tickers to `intraday_config.json`
- [ ] Run IC check on intraday proposals after 2 weeks
- [ ] Build `rh_executor.py` after paper signals validated
