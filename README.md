# Investment Pipeline — Quantitative Equity & Government-Contract Signal Engine

[![CI](https://github.com/ai-frankie/investment-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-frankie/investment-pipeline/actions/workflows/ci.yml)

A deterministic, backtested pipeline that ingests real-world financial data, scores equities on a multi-factor rubric, runs a transformer-based price forecaster (Kronos/PyTorch), and produces daily trade proposals against a tracked ledger.

> **Engineering thesis:** financial software fails on ambiguity and silent error. This pipeline is built for *reproducibility and auditability* — every proposal traces back to versioned inputs, a scoring rubric, and a backtest with an information-coefficient (IC) report.

**Engineering highlights:** ingests real-world financial data (SEC/EDGAR filings, USASpending government-contract awards, market data) · deterministic decision logic (model forecasts, rules decide) · walk-forward backtesting with big-data validation before any signal ships · append-only auditable state. Relevant to embedded-finance / BaaS data integrity.

---

## What it does

| Stage | Module | Purpose |
|---|---|---|
| Ingest | `kronos_data_fetcher.py`, `edgar_watcher.py`, `news_watcher.py` | Pull market data, SEC/EDGAR filings, and news signals |
| Score | `build_basket.py`, `learn_weights.py` | Multi-factor scoring rubric with learned factor weights |
| Forecast | `kronos_forecast.py` | Transformer (Kronos/PyTorch) price-horizon forecasting |
| Validate | `backtest.py`, `ic_report.py` | Walk-forward backtest + information-coefficient reporting |
| Execute | `ledger.py`, `daily_brief.py` | Proposal generation against a tracked position ledger |

## Architecture principles

- **Deterministic over clever** — the model forecasts; the *rules* decide. No black-box auto-execution.
- **Backtest discipline** — no factor ships without a walk-forward IC report. A horizon-alignment bug that inverted edge was caught and corrected by the backtest harness, not in production.
- **Auditable ledger** — every proposal, fill, and equity-curve point is written to append-only CSV state for reconstruction.

## Tests

```bash
pip install -r requirements.txt pytest
pytest test_pipeline.py -v
```

`test_pipeline.py` — **26 tests**, run in CI on every push. They pin the deterministic core the strategy rests on: performance statistics (CAGR / Sharpe / max-drawdown), the rank information coefficient, the BUY/HOLD/REDUCE entry-gate classifier and its regime gating, forecast-horizon math, the capped Kronos annualization, and cross-sectional factor z-scoring. Every input is synthetic and every expected value is hand-derived — no network, no data, no model inference. Writing these surfaced and fixed a latent crash in the factor-standardization path.

## Stack

Python · PyTorch (Kronos transformer) · pandas/numpy · SEC EDGAR + market-data APIs

## Status

Personal research pipeline, run on a daily schedule. Not investment advice.

---

## ⚠️ Repository hygiene (read before cloning/forking)

This repo is **code only**. Personal financial artifacts (account PDFs, filled forms, live position ledgers, run logs) are **excluded via `.gitignore`** and never committed. If you fork, supply your own `config.json` and data sources.
