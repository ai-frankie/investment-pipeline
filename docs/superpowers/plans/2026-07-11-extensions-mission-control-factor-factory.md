# Extensions Design — Phase I: Mission Control, Phase J: Factor Factory

Design-level spec (architect: Fable, 2026-07-11). Build later via runbook §5.6/§5.7 in MASTER-BLUEPRINT. Both extensions inherit all §1 design invariants — nothing here trades, activates itself, or touches decision logic.

---

## Phase I — Mission Control (observability + escalation)

**Why this extension:** every activation gate in the system routes through Frank reading evidence. Today that evidence is scattered across CSVs, logs, and stdout. The human interface is the binding constraint on the whole design — a 30-second daily read beats a 15-minute CSV dig, and degradation you don't see is degradation that compounds.

### I1 — Daily HTML report (`report_daily.py`)

- Output: `output/report/report_YYYYMMDD.html` + copy to `report_latest.html`. Single self-contained file (inline SVG/base64 charts, zero external requests, no server) — double-click to open, attach-able, Brain-pushable.
- Sections (top to bottom = most actionable first):
  1. **Verdict banner** — health_check summary line + any WARN/CRITICAL events since last report
  2. **Equity curve** — paper ledger, with fill markers; drawdown subplot; stale-price days shaded
  3. **Today's decisions** — proposals table: ticker, raw score, rank (adj_score), action, WHICH gate blocked it (skip_reason/regime_note/dispersion) — gates made visible, not just outcomes
  4. **Factor health strip** — per-factor rolling IC sparkline + FDR flag; anti-predictive factors highlighted red (vol_context-style finds surface in a day, not a quarter)
  5. **Signal lifecycle board** — every log-mode signal (insider, congress, dispersion, meta-P) with its accumulation progress toward activation threshold (n/60 run_dates etc.)
  6. **Data quality** — fetch failures, stale bars, cache fallbacks, run wall-clock vs norm
- Generation: appended to `run_scheduler.py` after the brief step; failure to render must never block the pipeline (same degrade-gracefully doctrine as daily_brief).
- Charts: matplotlib → inline SVG. No new heavy deps.

### I2 — Notification escalation policy

Replace ad-hoc plyer toasts with three tiers, one helper (`notify.py`), used by all modules:
- **INFO** — run completed (daily/intraday). Silent log line only by default; toast optional via config.
- **WARN** — data quality: stale prices, fetch fallback, brief fallback, dispersion of runtime >2× norm. Toast, non-blocking.
- **CRITICAL** — crashed scheduled run, ledger reconciliation mismatch, KILL/SHUT-OFF health verdict, (Phase F) order rejection or halt. Toast + write to `output/report/alerts.log` + banner in next daily report. CRITICAL alerts must be un-missable but never auto-act.

### I3 — Execution-quality loop (activates with Phase F)

- Every live order logs intended price (decision), submitted price, fill price → `ledger/intraday/live_orders.csv` (already in F design; I3 adds the analysis).
- Monthly slippage report: measured bps distribution vs the 10 bps assumed in backtest costs. If measured p50 > assumed, health_check gains a `COST-MODEL-STALE` verdict → Frank updates `--cost-bps` and re-runs OOS sweep. Closes the loop between assumed and real friction — most retail backtests never do this.

**Acceptance criteria (I-phase executor):** report renders from real data in <30s; pipeline completes even when report crashes; every module's prints routed through notify tiers; zero new decision logic anywhere.

---

## Phase J — Factor Factory (candidate signal assembly line)

**Why this extension:** the audit built an honest measurement loop; E-phase built log→validate→activate. The marginal cost of evaluating one more candidate signal is now near zero — the bottleneck is a standard intake protocol. This converts the system from "fixed 6 factors" into a pipeline that eats hypotheses and emits verdicts.

### J1 — Candidate protocol (the contract)

Every new factor enters ONLY via `factors/candidates.json` — one entry per candidate:

```json
{
  "name": "earnings_drift",
  "hypothesis": "positive surprise drift persists 3-10d post-announcement",
  "expected_ic_sign": "+",
  "formula_ref": "factors/earnings_drift.py",
  "status": "log",
  "added": "2026-07-11",
  "min_run_dates": 60
}
```

Rules (enforced by loader code, not convention):
- A candidate computes to a [0,1] score from data already fetched (or its own module with the standard retry/cache pattern). It is APPENDED as a factor_log column, weight 0 — logged, never in raw_score while `status: "log"`.
- `expected_ic_sign` declared BEFORE data is seen — a candidate whose measured IC is significant but opposite-signed is KILLED, not flipped (flipping after the fact = data mining).
- Promotion/kill: exact E4 criteria (FDR-significant, |IC| ≥ 0.03, ≥ min_run_dates), surfaced by health_check's existing verdict machinery — Factor Factory adds candidates to the same loop, no new judge.
- Hard cap: ≤ 5 candidates in log mode simultaneously (multiple-testing burden grows with every live test; FDR correction spans candidates too).

### J2 — Seed backlog (designed candidates, build order by data-readiness)

| Candidate | Formula sketch | Expected sign | Data |
|---|---|---|---|
| `earnings_drift` | sign(last surprise) × decay(days since announcement, 10d half-life) | + | days_to_earnings logged; surprise via yfinance earnings history |
| `gap_reversal` | −zscore(overnight gap) when \|gap\| > 1.5× ATR20 | + (fade) | daily OHLC, have it |
| `sector_rel_strength` | ticker 20d return − its sector ETF 20d return, rank-scaled | + | sector ETFs already in universe (XLK/XLF/XLE/XBI) |
| `high_52w_proximity` | close / 252d max, rank-scaled | + | daily closes, have it |
| `st_reversal_5d` | −rank(5d return) | + (reversal) | daily closes, have it |
| `volume_confirm` | corr(sign(daily ret), volume zscore, 10d) | + | daily OHLCV, have it |
| `seasonality_dow` | day-of-week/turn-of-month historical mean | ~0 expected | ⚠ include as a NEGATIVE CONTROL — a factor expected to fail; if it "passes," the harness is broken, not the calendar magical |

The negative control is deliberate: a measurement loop that can't fail a placebo can't be trusted to pass a real signal.

### J3 — Intake cadence

- Max 2 new candidates per month (keeps FDR family small, keeps run_date accumulation meaningful per candidate).
- Each candidate's lifecycle appears on the Mission Control signal board (I1 §5) automatically.
- Kill fast doctrine: a candidate at min_run_dates with |IC| < 0.02 is retired, entry moved to `factors/graveyard.json` with its measured IC — the graveyard is the lab notebook; dead hypotheses are paid-for knowledge and prevent re-testing the same idea next year.

**Acceptance criteria (J-phase executor):** loader enforces protocol rules in code; one candidate (`high_52w_proximity` — simplest, data in hand) fully wired as reference implementation; health_check + report show it; raw_score provably unchanged (test: scores identical with candidates present vs absent while status=log).

---

## Sequencing

- Phase I: buildable immediately after Phase E lands (report needs E1's columns and E8's health_check to have content). I3 waits for F.
- Phase J: buildable after E lands (needs E4's promotion machinery + I1's board for visibility). First candidate wired same session; backlog drips at ≤2/month.
- Neither phase blocks nor is blocked by F/G — parallel tracks.
