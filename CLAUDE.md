# investment-pipeline — Project Context

## What This Is
Local-first quant scoring system for a Fidelity Rollover IRA ($126k).
Python 3.14, PyTorch CPU, zero cloud dependencies, zero paid data subscriptions.
Two pipelines: daily (Fidelity IRA) + intraday (Robinhood agentic paper mode).

**Purpose:** Generate daily/hourly trading signals using technical factors + macro gates. Paper-trade to validate edge before live execution.

**Edge thesis:** Momentum (7-bar returns) + trend alignment (EMA ratio) + volatility context (RV20) identify mean-reversion setups with positive expected return. Kronos (daily, 1d/10-candle) adds price-forecast layer. Intraday gates (PDT, VIX, macro, news) reduce false positives. Target Sharpe ≥0.70.

---

## Architecture

**Data flow:**
```
yfinance (1h/1d bars) → compute_score() [4-6 factors] → apply gates [PDT/macro/VIX/daily/news]
  → BUY/HOLD/REDUCE/SKIP action → paper ledger (positions.csv + equity_curve.csv)
  → daily_brief.py (Ollama LLM or data fallback) → output/proposals_*.csv
```

**Pipeline layers:**
1. **Factor computation** — momentum, volume, trend, volatility
2. **Kronos (daily only)** — transformer-based price forecast (10-candle horizon)
3. **Gating** — PDT budget, macro calendar, VIX, daily regime, news sentiment
4. **Position management** — flat sizing (intraday) / optimized (daily)
5. **Reporting** — CSV proposals, brief, equity curve, trade ledger

---

## Dependencies

**Python:** 3.14 (required for torch 2.12.1+cpu)

**Packages:**
- `yfinance` — market data fetch
- `pandas`, `numpy` — data manipulation
- `torch` 2.12.1+cpu — Kronos inference
- `apscheduler`, `plyer` — scheduling + Windows notifications
- `transformers` — FinBERT news sentiment
- `scikit-learn` — standardization, preprocessing
- `pypfopt` 1.5+ — Efficient Frontier + Ledoit-Wolf shrinkage
- `requests` — FRED/USASpending HTTP

**External:**
- Ollama (llama3.1:8b) — local LLM for daily brief (optional; data fallback if down)
- Windows Task Scheduler — daily/intraday cron
- FRED API (free, no key) — HY OAS, NFCI macro gates
- USASpending.gov (free, no key) — gov contract signals

**Install:**
```bash
pip install yfinance pandas numpy torch apscheduler plyer transformers scikit-learn pypfopt requests
# Ollama: download from ollama.ai, run `ollama pull llama3.1:8b`
```

---

## Quick Start

**Manual runs:**
```bash
# Python 3.14 only
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe pipeline.py           # Daily scan
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe intraday_pipeline.py  # Hourly scan
C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe intraday_pipeline.py --force-close  # End-of-day close
```

**Automated:**
- Daily: Task Scheduler "Investment Pipeline Scheduler" → 4:30pm ET (runs `run_scheduler.py --now`)
- Intraday: Task Scheduler "Intraday Pipeline" → 9:45, 11:00, 12:30, 14:00, 15:25 ET

**Check outputs:**
- Daily proposals: `output/proposals_YYYYMMDD_*.csv`
- Intraday proposals: `output/intraday/proposals_intraday_YYYYMMDD.csv`
- Daily brief: `output/briefs/brief_YYYYMMDD.md`
- Equity: `ledger/equity_curve.csv`, `ledger/intraday/positions.csv`

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

Kronos IC — **superseded, do not cite +0.145.** Re-measured twice during the
2026-07-11 audit fix: Gate A (non-overlapping windows only) → +0.191; post-C1
(median real sampled path, removes a mean-of-paths smoothing artifact) →
**+0.061**, hit rate 50% (coin flip), avg realized return on SIGNAL-fired
bars now **negative**. See `notes/2026-07-11-revalidation-results.md`
(Addendum). This is Frank's call on whether the daily Kronos signal is still
worth carrying — plumbing fixes alone won't move this number further.

**Alt-data modifiers (congress/insider) — measured log-mode (Phase E, E4):**
`congress_mode`/`insider_mode` in `config.json` are `"off"` | `"log"` | `"active"`.
Both ship `"log"` as of 2026-07-19 (D1: insider was silently `active` at +0.10 on
a [0,1] score — flips HOLD→BUY at 0.60 — and had never been IC-measured; that's
the biggest unvalidated lever in the scorer). `"log"` still fetches + computes
`congress_mod`/`insider_mod` and logs them to `factor_log.csv` every run (plus
`mods_applied`, which mode(s) actually moved `raw_score` that row) but does not
add them to the score. Both are now in `ic_report.py`'s `FACTORS` list and get
IC + FDR significance like every other factor.
**Promotion rule:** `log` → `active` only when FDR-significant IC over ≥60
run_dates with `|IC| ≥ 0.03`. **Demotion:** back to `log` if significance is
lost on 2 consecutive monthly checks. See
`docs/superpowers/plans/2026-07-11-phase-e-alpha-upgrades.md` (Task E4).

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

**Core:**

| File | Purpose |
|---|---|
| `pipeline.py` | Daily scorer (6 factors + optimizer) |
| `intraday_pipeline.py` | Intraday scorer (4 factors, paper mode) |
| `run_scheduler.py` | Called by Task Scheduler; runs pipeline + daily_brief |
| `kronos_forecast.py` | Kronos inference wrapper |
| `ledger.py` | Daily paper ledger + equity curve |
| `news_watcher.py` | FinBERT news veto (7 categories) |
| `daily_brief.py` | Generate daily brief (Ollama LLM or data fallback) |

**Config:**

| File | Purpose |
|---|---|
| `config.json` | Daily pipeline (16 tickers, 1d/10-candle, 6 factors) |
| `intraday_config.json` | Intraday pipeline (48+ tickers, 1h, 4 factors) |
| `macro_calendar.json` | FOMC/CPI/NFP dates — verify yearly |

**Support:**

| File | Purpose |
|---|---|
| `backtest.py` | Walk-forward IC backtest (`--mode kronos --interval 1h --horizon 12`) |
| `ic_report.py` | IC analysis utility |
| `learn_weights.py` | Factor weight optimization (not actively used) |
| `edgar_watcher.py`, `quiver_congress_watchlist.py` | Congress/SEC signal watchers (log-mode, unmeasured — see Daily Pipeline) |
| `usaspending_watcher.py` | Gov contract signals |

**Dead / Experimental:**

| File | Purpose | Status |
|---|---|---|
| `debug.py`, `make_brief.py`, `make_brief_fixed.py` | Early-stage brief experiments | Archive or delete |
| `research_script.py`, `research_script_final.py` | Ad-hoc research | Archive or delete |
| `robinhood_fetcher.py` | RH data fetch (standalone, not wired) | Archive or delete |
| `build_basket.py` | Portfolio construction (superseded) | Archive or delete |

**Kronos/ (model library):**
- `model/` — trained Kronos model weights
- `finetune/` — fine-tuning scripts
- `tests/` — unit tests
- `webui/` — debug web interface
- `examples/` — example notebooks
- `figures/` — plots from training

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

## Changelog

| Date | File | Change | Reason | Commit |
|---|---|---|---|---|
| 2026-07-19 | `pipeline.py`, `intraday_pipeline.py`, `config.json`, `ic_report.py`, `learn_weights.py`, `test_scoring.py`, `test_meta.py`, `build_meta_dataset.py` (new), `health_check.py` (new) | Phase E alpha upgrades (E1-E8, executed E1→E4→E3→E2→E7→E5→E6→E8): log kronos path_dispersion/n_paths + intraday news_sent; congress/insider modifiers to measured log-mode (D1); cross-sectional rank as adj_score + top-quartile BUY filter + 80-name universe + bulk fetch (D2 — mechanical rule now yields ~80, not the ~60 estimated, see Pending); kronos path-dispersion conviction gate in log mode + num_paths 7 (D3); meta-label dataset extractor (build_meta_dataset.py, NOT READY n=0 today); per-regime IC diagnostics + gated `learn_weights.py --regime`; edge-scaled vol targeting shipped dormant behind kelly_fraction=0; weekly health_check.py (kill/promote/activate recommendations, never self-acts) | 2026-07-11 phase-e-alpha-upgrades plan, Branch 2 posture (post-C1 Kronos IC +0.061 is thin; E3 rank scoring + E4 alt-data are the primary alpha hopes) | 8266205, 4c20f6c, 722610c, be79877, 627467f, 1a9ab63, c26ed1c, (E8 TBD) |
| 2026-07-17 | `intraday_pipeline.py`, `test_intraday.py` | Addendum X1+X2: all-tickers-errored scan now fails loud ([SCAN-FAILURE] + plyer + exit 1, gated scans still exit 0); force-close falls back to latest 1h close and RETAINS unpriced positions instead of wiping them ([FORCE-CLOSE-FAILED] + exit 1) | Jul 15–17 silent-fail incident: 2 days of scans produced nothing at exit 0; Jul 15 force-close logged AAPL/MSFT with NaN PnL then cleared positions | db0b237, 43e5c41 |
| 2026-07-17 | `pipeline.py`, `test_pipeline.py`, `test_scoring.py`, `backtest.py`, `intraday_pipeline.py`, `usaspending_watcher.py`, `edgar_watcher.py`, `quiver_congress_watchlist.py`, `robinhood_fetcher.py`, `run_scheduler.py`, `config.json`, archive/ | Audit-fix plan Phase C+D (C1–D5): Kronos median-real-path + parameter-aware cache, saturating mu/edge scores, config VIX ceiling, gate observability, watcher retry+cache-partial-failure fixes, RH tz fix, real Ollama readiness poll, shared BUY/HOLD constants, archived 6 dead scripts | 2026-07-11 audit fix plan, resume-and-continue dispatch | d1627a2, 85ca3a6, 8664705, 2aba756, 702ef25, 65586ca, 506eb2c |
| 2026-07-11 | `ic_report.py`, `pipeline.py`, `intraday_pipeline.py`, `backtest.py`, `learn_weights.py` | Audit-fix plan Phase A (A1–A6): honest IC measurement (next-session-open fill anchor, dedup, FDR-corrected significance), same-day factor_log dedup, NaN-safe vol_context, trend-gate/scorer band alignment, backtest transaction costs + OOS split + non-overlapping kronos windows, learn_weights negative-factor drop + exact purge | 2026-07-11 multi-agent audit — 42 verified defects, measurement integrity first | 39950e2, 06cc6ee, 75a6890, c86e068, 779e644, f22267d |
| 2026-07-11 | `ledger.py`, `intraday_pipeline.py`, `intraday_config.json`, `ledger/archive/` | Audit-fix plan Phase B (B1–B6): session-aware fills (no weekend/stale fills), atomic crash-safe writes, PDT tracker persistence + corrupt-state recovery, flat position sizing + dollar PnL, explicit ET tz + entry-time gate + idempotent appends, pre-fix ledgers archived and paper-validation clock restarted | Paper ledger was silently non-functional (inert PDT gate, unsized positions, weekend fills) | e1e602c, 6e1ed14, 99a9eac, 3a94e7d, 842c706, (B6: archive-only, `ledger/`+`notes/` are gitignored, no commit) |
| 2026-07-10 | `CLAUDE.md` | Added Architecture, Dependencies, Quick Start, expanded Key Files, Investment Thesis, File Inventory | Hermes audit — improve onboarding + document dead scripts | TBD |
| 2026-07-10 | `intraday_pipeline.py` | **BUGS IDENTIFIED** (qty=None line 429, force-close cascades, no cash tracking) | Hermes audit Step 3 — intraday paper trading non-functional | TBD |
| 2026-07-10 | `requirements.txt` | **CREATED** (new file) | Hermes audit Step 4 — missing dependency lock | TBD |
| 2026-07-10 | `README.md` | **CREATED** (new file) | Hermes audit Step 4 — missing project readme | TBD |
| 2026-07-06 | `daily_brief.py` | Added `fallback_brief()` — data-only brief when Ollama unreachable | Brief was stale Jun 9→Jul 6; Ollama down, no fallback | f243b3e |
| 2026-07-06 | `run_scheduler.py` | Added `ensure_ollama()` to auto-start Ollama before brief | Ollama needed but not running on schedule | f243b3e |
| 2026-07-10 | `intraday_pipeline.py` | Audited: found qty sizing gap (hardcoded None, no position sizing logic) | closed_trades.csv missing qty/PnL; rh_executor.py still TODO | — |

**Known issues:**
- SPY/QQQ not in daily proposals (tracked but not scored — benchmark only?).
- BAH forecast anomaly (+33.94% on Jul10 scan) — likely data artifact.
- ~~Intraday position sizing: qty hardcoded to None~~ — **resolved 2026-07-11**, see B4 (flat sizing, paper_equity x 40%).
- Kronos daily IC no longer supports "usable edge" at face value post-C1 honest re-check (+0.061, hit rate 50%) — see Daily Pipeline section above and `notes/2026-07-11-revalidation-results.md`.

**Known limitations (by design, not bugs):**
- Intraday daily-gate only covers the 16-ticker daily universe (`config.json`) — a ticker in `intraday_config.json`'s larger universe but not in the daily config gets `daily_action: "N/A"` (visible via the `[GATE] daily gate covers N/M tickers` scan-start print) and is never REDUCE-skipped by that gate. This is a coverage gap, not a bug — expanding the daily universe is a separate decision.
- The regime gate (`check_regime` / VIX ceiling / trend-flat veto) is an **entry gate only** — it blocks new BUYs but never force-halves an existing position's score or forces an exit. See `action_label`'s docstring for the de-conflicted risk-stack rationale (regime gates, score ranks, vol targeting sizes — no double-counting).

---

## Pending

- [ ] Fund Robinhood agentic ••••6090 (~$2k, Frank's task)
- [ ] Add 100+ tickers to `intraday_config.json` tickers array
- [ ] Wire `health_check.py` (Phase E task E8) to the Friday 4pm weekly audit ritual — `run_scheduler.py` hook or Task Scheduler, Frank's call. Run manually until then: `python health_check.py` (read-only, writes `output/health/health_YYYYMMDD.txt`).
- [ ] Review Phase E's universe expansion (16 -> 80 tickers in `config.json`, see 2026-07-19 changelog entry) — D2 was approved at ~60; the mechanical rule now yields ~80 since `intraday_config.json` grew since the estimate. Live-verified runtime is acceptable (~10-16 min at num_paths=7) but worth a deliberate look, not a default acceptance.
- [ ] Run IC check on intraday proposals after 2 weeks
- [ ] Build `rh_executor.py` after paper signals validated
- [ ] Run `/wrapup` and push to Brain at session end
- [ ] Standing: after `/wrapup` writes session notes, run `/graphify --update` — keeps knowledge graph + Obsidian vault (`graphify-out/obsidian/`) current; `.graphifyignore` re-includes gitignored `notes/` for local graphing only