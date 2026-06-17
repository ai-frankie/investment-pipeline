# Investment Pipeline — Full Setup Overview

*Built by Frank + Claude Code, June 2026. Local-first quant system for a six-figure
rollover IRA. Everything runs on a Windows 11 desktop: Python 3.14, PyTorch CPU,
zero cloud dependencies, zero paid data subscriptions.*

---

## What this is

An automated daily stock-scoring system. Every weekday at 4:30pm ET it forecasts
prices, scores ~16 tickers across 6 factors, checks 4 layers of risk gates,
optimizes position sizes, logs every decision for later statistical validation,
and writes a plain-English brief using a local LLM. A human approves every trade —
the system is advisory by design, never autonomous with money.

## Architecture (data → forecast → signal → risk → sizing → ledger)

```
yfinance OHLCV ──> Kronos foundation model (3 sampled paths per ticker)
                        │
USASpending.gov ──┐     ▼
SEC EDGAR Form 4 ─┼─> 6-FACTOR SCORE (0-1) ──> entry gates ──> PyPortfolioOpt
FinBERT headlines ┘     │                        │               max-Sharpe + caps
FRED macro data ────────┘                        │                    │
                                                 ▼                    ▼
                                          BUY / HOLD / REDUCE   vol-targeted sizing
                                                 │
                                          paper + real ledger, decision log,
                                          IC dashboard, Ollama daily brief
```

## The six factors (equal-weighted now, learned weights later)

1. **Forecast edge** — Kronos (open-source OHLCV foundation model, NeoQuasar on
   HuggingFace) samples 3 independent price paths; median forward return is
   measured against a volatility-scaled threshold (1σ move over the forecast
   horizon — not a fixed %, which silently kills the factor on low-vol names).
2. **Path consistency** — P(up): fraction of sampled paths ending positive.
3. **Volatility context** — 20-day realized vol vs its 1-year median.
4. **Trend alignment** — EMA20/EMA50 separation band.
5. **Long-term quality** — 3-year annualized return proxy.
6. **Gov-contract signal** — USASpending.gov API: new awards/modifications for
   defense names (PLTR, CACI, SAIC, BAH, LDOS, RTX, LMT, NOC, GD). Free, 3-7 day
   lag on federal contract awards.

Plus bounded score modifiers:
- **SEC Form 4 insider clusters** (official EDGAR API, no key): open-market P-code
  buys only, 10b5-1 planned trades excluded; 3+ insiders/$100k/30 days = +0.10.
- **Congressional trades** (Quiver API): net politician buys/sells = ±0.05.

## The risk stack (each layer does ONE job — no double-counting)

- **Regime gate** (entry only): VIX ceiling, realized-vol spike, trend-flatness,
  plus macro credit conditions from FRED (high-yield spreads vs 1y median, NFCI).
- **Event blackouts**: no new entries within 3 days of earnings or 1 day of
  FOMC/CPI/NFP — foundation models can't see gap events coming.
- **News veto**: FinBERT (finance-tuned BERT, runs locally) scores headlines with
  recency weighting; 7 hard event classes (fraud, investigation, guidance cut,
  C-suite exit, solvency, deal break, adverse verdict) block entry regardless of
  sentiment sign.
- **Vol targeting** (sizing only): gross exposure scaled to a 10% annualized
  portfolio vol target, 0.25×–1.5× bounds, on top of max-Sharpe weights with
  20% per-position caps (Ledoit-Wolf shrinkage covariance).

## The validation discipline (what separates this from vibes)

- **Walk-forward backtester**, two modes: fast factor backtest (no look-ahead,
  weekly rebalance, vs buy-and-hold) and real Kronos inference replayed at
  historical as-of dates → information coefficient + hit rate.
- **Decision log**: every run appends all factor values, gates, prices, and
  actions to an append-only CSV (schema-drift-guarded).
- **IC dashboard**: joins realized forward returns onto the log; per-factor
  Spearman IC plus a model-decay alarm (rolling IC < 0.02 for 8 weeks → stop
  trusting the forecaster).
- **Weight learner**: pooled panel ridge regression with purged/embargoed
  walk-forward CV (labels overlap, so naive CV leaks); refuses to fit under 100
  observations; auto-rejects weak fits after multiple-testing adjustment.
- **Paper + real ledger**: proposals fill at next open with 10bps slippage each
  way; real fills seeded at actual cost basis; daily mark-to-market equity curve
  reconciles to the brokerage statement.

## Files

| File | Job |
|---|---|
| `pipeline.py` | Daily scoring engine: factors, gates, modifiers, optimizer, decision log |
| `kronos_forecast.py` / `kronos_data_fetcher.py` | Foundation-model forecasts, cached model, multi-path sampling, as-of mode for backtests |
| `backtest.py` | Factor + Kronos walk-forward validation |
| `ledger.py` | Paper/real trade ledger, fills, slippage, equity curve |
| `ic_report.py` | Factor IC + model-decay dashboard |
| `learn_weights.py` | Leak-proof ridge factor-weight learning |
| `edgar_watcher.py` | SEC Form 4 insider-cluster ingestion (stdlib XML, official endpoints) |
| `usaspending_watcher.py` | Federal contract award signals |
| `quiver_congress_watchlist.py` | Congressional trading signals |
| `news_watcher.py` | FinBERT sentiment + hard event veto |
| `build_basket.py` | Universe expansion: $10M ADV floor, sector caps, correlation clustering |
| `daily_brief.py` | Local Ollama LLM writes the daily commentary → pushed to NotebookLM |
| `run_scheduler.py` | APScheduler daily driver + Windows toast notifications |
| `config.json` / `macro_calendar.json` | All tunables and event dates in one place |

## Why Claude Code specifically (the honest pitch)

This was built in conversational sessions where the assistant **did the work
directly on the machine**, not by generating code blocks to copy-paste:

- **Reads and edits real files across a 14-file codebase** — surgical diffs, not
  full-file regeneration that loses your changes.
- **Runs and verifies everything live**: it executed the pipeline against real
  APIs mid-session, caught its own bugs (an infeasible optimizer constraint, a
  corrupted CSV schema, a dead data source returning 403s), and fixed them before
  calling anything done. ChatGPT hands you code; this hands you *tested* code.
- **Long-running background tasks**: a 31-ticker model scan ran in the background
  while we kept working; it reported back when done.
- **Persistent memory across sessions**: it remembers the project state, pending
  actions, and key dates between sessions without re-explaining anything.
- **Orchestrates the whole local stack**: it drove Ollama, NotebookLM CLI, SEC
  EDGAR, FRED, and Windows Task Scheduler — gluing tools together instead of
  describing how you might.
- **Honest engineering posture**: it pushed back on "hot trades" and point
  predictions, refused to fake validation, and built the discipline layers
  (blackouts, decay alarms, purged CV) that prevent self-deception with real money.

The whole build — forecasting, signals, risk stack, validation, ledger, local-LLM
offload — took a handful of sessions. Total recurring cost: $0/month in data and
infrastructure.

*Not investment advice. The system proposes; the human decides.*
