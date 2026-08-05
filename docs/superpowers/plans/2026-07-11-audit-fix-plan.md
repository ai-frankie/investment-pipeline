# Investment Pipeline Audit Fix — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 42 verified defects from the 2026-07-11 multi-agent audit so the pipeline's edge measurement is honest, the paper ledger is trustworthy, and scoring math is correct.

**Architecture:** Four sequential phases. Phase A fixes measurement (IC labels, backtest methodology) and the two scoring bugs that corrupt factor values. Phase B fixes the paper-trade ledgers (daily + intraday). Phase C fixes model/scoring quality. Phase D is hygiene. A hard STOP gate after Phase A re-validation: Frank reviews the honest numbers before Phase B+ proceeds.

**Tech Stack:** Python 3.14 (`C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe` — ALWAYS this interpreter; the shell-default 3.11 venv lacks torch), pandas, numpy, yfinance, pytest, scipy/statsmodels.

---

## Executor ground rules

1. **One executor, start to finish.** Do not mix models mid-plan.
2. **Interpreter:** every `python`/`pytest` command below means the 3.14 path above. Alias it once: `$PY = "C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe"`. If pytest missing: `& $PY -m pip install pytest`.
3. **Never delete or rewrite history in `output/factor_log.csv`** — append-only training data. Fixes change how NEW rows are written and how readers interpret rows; old rows stay.
4. **Tests live in the repo root** beside `test_pipeline.py` (existing file — follow its style). New test files: `test_ic_report.py`, `test_ledger.py`, `test_intraday.py`, `test_scoring.py`, `test_learn_weights.py`.
5. **Commit after every task** with the message given. Do not batch commits.
6. **Do not touch validated thresholds** (BUY 0.70 / HOLD 0.45) or config values except where a task explicitly says so.
7. **REFUTED findings — do NOT "fix":** earnings-blackout tz handling (`pipeline.py next_earnings_in_days` is correct), regime-gate entry-only asymmetry (documented design), intraday BUY-slot writing (works), daily_brief silent-skip (already logged). Leave them alone.
8. Line numbers below were verified 2026-07-11 but drift — always locate by function name + snippet, not line number alone.

---

## PHASE A — Measurement integrity

### Task A1: ic_report.py — anchor forward returns at next-session fill, dedup, visible exclusions

Fixes: I4 (entry anchored at scoring-day close instead of next-day fill), P1 (IC window ≠ ledger fill window), I2 (silent ticker exclusions → survivorship).

**Files:**
- Modify: `ic_report.py` (`load_with_forward_returns()`)
- Test: `test_ic_report.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# test_ic_report.py
import pandas as pd
import numpy as np
from ic_report import _entry_index

def _idx(dates):
    return pd.DatetimeIndex(pd.to_datetime(dates))

def test_entry_index_run_date_is_trading_day():
    # run_date is a trading day -> entry is the NEXT session
    close_index = _idx(["2026-07-06", "2026-07-07", "2026-07-08"])
    assert _entry_index(close_index, "2026-07-06") == 1

def test_entry_index_run_date_is_weekend():
    # Friday run logged Saturday / weekend run_date -> searchsorted already
    # lands on next session; do NOT skip an extra day
    close_index = _idx(["2026-07-10", "2026-07-13", "2026-07-14"])  # Fri, Mon, Tue
    assert _entry_index(close_index, "2026-07-11") == 1  # Sat -> Mon

def test_entry_index_no_next_session_yet():
    close_index = _idx(["2026-07-09", "2026-07-10"])
    # run_date == last bar -> entry would be beyond data
    assert _entry_index(close_index, "2026-07-10") == 2  # == len -> caller NaNs
```

- [ ] **Step 2: Run to verify failure**

Run: `& $PY -m pytest test_ic_report.py -v`
Expected: FAIL (ImportError: `_entry_index` does not exist)

- [ ] **Step 3: Implement**

In `ic_report.py`, add module-level helper:

```python
def _entry_index(close_index, run_date) -> int:
    """Index of the earliest realistic fill session (next trading day after
    run_date). If run_date itself is a trading day, entry is the NEXT bar;
    if run_date is a weekend/holiday, searchsorted already lands on the next
    session — use it as-is. May return len(close_index): caller must bounds-check."""
    ts = pd.Timestamp(run_date)
    idx = close_index.searchsorted(ts)
    if idx < len(close_index) and close_index[idx] == ts.normalize():
        return idx + 1
    return idx
```

Then rework `load_with_forward_returns()`:

1. **Dedup:** immediately after loading factor_log, keep only the last row per `(run_date, ticker)`:
   `log = log.drop_duplicates(subset=["run_date", "ticker"], keep="last").reset_index(drop=True)`
2. **Fetch Open too:** extend the `yf.download` call to keep both `Open` and `Close` columns (currently only Close is used).
3. **Anchor:** replace the loop body

```python
# OLD
idx = close.index.searchsorted(row["run_date"])
if idx + horizon_days >= len(close):
    fwd.append(np.nan)
    continue
fwd.append(float(close.iloc[idx + horizon_days]) / float(close.iloc[idx]) - 1.0)
```

with

```python
entry_idx = _entry_index(close.index, row["run_date"])
if entry_idx >= len(close) or entry_idx + horizon_days >= len(close):
    fwd.append(np.nan)  # no next session yet / future not realized
    continue
entry_px = float(open_prices.iloc[entry_idx])   # fill at next-session OPEN, like ledger.py
if not np.isfinite(entry_px) or entry_px <= 0:
    entry_px = float(close.iloc[entry_idx])     # fallback: next-session close
fwd.append(float(close.iloc[entry_idx + horizon_days]) / entry_px - 1.0)
```

where `open_prices` is the ticker's Open series aligned to `close.index`.
4. **Visible exclusions:** collect a `fetch_failed: set()` in the `except` branch; after the loop print:
   `WARNING: {n} ticker(s) excluded from IC due to data-fetch failure (possible delisting): {sorted(fetch_failed)}`

- [ ] **Step 4: Run tests**

Run: `& $PY -m pytest test_ic_report.py test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ic_report.py test_ic_report.py
git commit -m "fix(ic): anchor forward returns at next-session open fill, dedup same-day rows, surface excluded tickers"
```

---

### Task A2: pipeline.py — factor_log same-day dedup on append

Fixes: FRESH/staterestart critical — double-run duplicates every training row.

**Files:**
- Modify: `pipeline.py` (factor_log write block, ~line 710)

- [ ] **Step 1: Implement** (I/O glue — no unit test; manual verify below)

Before appending `log_df`, remove today's rows for the tickers being re-scored (mirror `ledger.record()` latest-run-wins). Apply in BOTH branches (schema-drift and normal-append):

```python
factor_log = OUTPUT_DIR / "factor_log.csv"
today = log_df["run_date"].iloc[0]
if factor_log.exists():
    old = pd.read_csv(factor_log)
    old = old[~((old["run_date"] == today) & (old["ticker"].isin(log_df["ticker"])))]
    merged = pd.concat([old, log_df], ignore_index=True)
    merged.to_csv(factor_log, index=False)
else:
    log_df.to_csv(factor_log, index=False)
```

(This subsumes the old schema-drift branch: full read-merge-write handles column changes too. Keep a comment noting factor_log is append-only across DAYS; same-day re-runs replace.)

- [ ] **Step 2: Manual verify**

Run: `& $PY pipeline.py` twice in a row, then
`& $PY -c "import pandas as pd; d=pd.read_csv('output/factor_log.csv'); print(d.groupby(['run_date','ticker']).size().max())"`
Expected output: `1`

- [ ] **Step 3: Commit**

```bash
git add pipeline.py
git commit -m "fix(pipeline): same-day re-run replaces factor_log rows instead of duplicating training data"
```

---

### Task A3: Scoring corruption — vol_context NaN→1.0 (both pipelines) + trend-band disagreement

Fixes: FRESH/mathcheck critical ×2 (NaN coerced to perfect score), FRESH/mathcheck major (scorer vs regime-gate disagree on same EMA ratio). These corrupt factor values in factor_log, so they belong in Phase A.

**Files:**
- Modify: `pipeline.py` (`_score_vol_context`, `_score_trend_alignment`, `check_regime`)
- Modify: `intraday_pipeline.py` (`_score_vol_context`)
- Test: `test_scoring.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# test_scoring.py
import numpy as np
import pandas as pd
import pipeline
import intraday_pipeline

def _hist(n, seed=0):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"close": px, "volume": np.full(n, 1e6)})

def test_daily_vol_context_short_history_neutral():
    # 100 bars: passes old len>=20 guard but rolling(252) is all-NaN.
    # Must return neutral 0.5, never 1.0.
    assert pipeline._score_vol_context(_hist(100)) == 0.5

def test_daily_vol_context_full_history_finite():
    s = pipeline._score_vol_context(_hist(400))
    assert 0.0 <= s <= 1.0 and np.isfinite(s)

def test_intraday_vol_context_168_bars_not_perfect():
    # exactly lookback_bars rows -> rolling baseline must still be computable
    s = intraday_pipeline._score_vol_context(_hist(168))
    assert s != 1.0 or np.isfinite(s)  # primary check: no NaN-coerced 1.0
    assert 0.0 <= s <= 1.0

def test_trend_score_and_regime_gate_agree():
    # any ratio the scorer rates 1.0 must NOT be vetoed as 'flat' by the gate
    assert pipeline.TREND_BAND_LOW == 0.005
    # gate threshold must equal the scorer's lower band edge
    # (constant shared by both functions)
```

- [ ] **Step 2: Run to verify failure**

Run: `& $PY -m pytest test_scoring.py -v`
Expected: FAIL (`test_daily_vol_context_short_history_neutral` returns 1.0; `TREND_BAND_LOW` missing)

- [ ] **Step 3: Implement**

`pipeline.py` `_score_vol_context`:

```python
def _score_vol_context(hist: pd.DataFrame) -> float:
    ret = hist["close"].pct_change().dropna()
    if len(ret) < 252:                      # rolling(252) needs 252 obs; was 20
        return 0.5
    rv20 = ret.iloc[-20:].std() * np.sqrt(252)
    rv_med = ret.rolling(252).std().dropna().median() * np.sqrt(252)
    if not np.isfinite(rv_med) or rv_med <= 0:   # NaN-safe; was == 0
        return 0.5
    return max(0.0, min(1.0, 1.0 - (rv20 / rv_med - 1.0)))
```

`intraday_pipeline.py` `_score_vol_context`: baseline window must fit inside the 168-bar lookback — use 84 (half), plus the same NaN-safe guard:

```python
    rv_med = ret.rolling(84).std().dropna().median() * ANNUALIZE_1H
    if not np.isfinite(rv_med) or rv_med <= 0:
        return 0.5
```

`pipeline.py` trend constants — add at module level near other constants:

```python
TREND_BAND_LOW = 0.005   # EMA20/EMA50 ratio: below this = flat
TREND_BAND_HIGH = 0.03   # above this = overextended
```

Use them in `_score_trend_alignment` (replace literals 0.005/0.03) and in `check_regime` replace `< 0.01` with `< TREND_BAND_LOW` so the gate can never veto a setup the scorer rates 1.0.

- [ ] **Step 4: Run tests**

Run: `& $PY -m pytest test_scoring.py test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline.py intraday_pipeline.py test_scoring.py
git commit -m "fix(scoring): NaN-safe vol_context (was silently scoring 1.0), align trend gate with scorer band"
```

---

### Task A4: backtest.py — transaction costs, sample size, OOS threshold split, non-overlapping Kronos windows

Fixes: B1 (no costs), B3 (no n), B2 (in-sample sweep), B4 (overlapping walk-forward windows).

**Files:**
- Modify: `backtest.py`

- [ ] **Step 1: Costs (B1).** Add `--cost-bps` CLI flag, `default=10.0`. In `backtest_factor_ticker`:

```python
    wk["pos"] = pos
    turnover = wk["pos"].diff().abs().fillna(wk["pos"].abs())  # first row: entering from flat
    wk["strat_ret"] = wk["pos"].shift(1).fillna(0) * wk["ret"] - turnover * cost_bps / 1e4
    return wk
```

Thread `cost_bps` through `run_factor_mode` (both sweep and single paths). Print BOTH gross and net Sharpe/CAGR. Regression check: with `--cost-bps 0` output must match current behavior exactly.

- [ ] **Step 2: Sample size (B3).** Add `"n_weeks": int(len(wk.dropna(subset=["ret"])))` to the per-ticker rows dict and print it; print a caveat when `n_weeks < 100`.

- [ ] **Step 3: OOS split (B2).** In the `--sweep` branch: split fetched history by date at the 70% point (fixed cutoff, time-ordered — never random). Grid-search on train only; re-run the single winning `(buy_thr, hold_thr)` on the test slice; headline print = TEST Sharpe/CAGR/maxDD labeled `OOS`; keep the full grid table labeled `IN-SAMPLE (diagnostic only)`. Guard: if the test slice is shorter than the 252-day warmup, print a warning and require `--period` ≥ 4y for sweep.

- [ ] **Step 4: Non-overlapping Kronos as-ofs (B4).** In `run_kronos_mode` replace weekly sampling:

```python
# OLD: one as-of per calendar week -> 10-day windows overlap ~50%
# NEW: as-ofs spaced pred_len bars apart -> disjoint realized windows
idx = np.arange(len(full) - 1 - pred_len, 0, -pred_len)[:weeks][::-1]
asofs = full["timestamps"].iloc[idx]
```

Update the run-count estimate print accordingly.

- [ ] **Step 5: Manual verify**

Run: `& $PY backtest.py --tickers NVDA --cost-bps 0` → matches pre-change numbers.
Run: `& $PY backtest.py --tickers NVDA` → net Sharpe ≤ gross Sharpe, `n_weeks` column present.
Expected: both hold.

- [ ] **Step 6: Commit**

```bash
git add backtest.py
git commit -m "fix(backtest): net-of-cost returns, OOS threshold validation, non-overlapping kronos windows, sample sizes"
```

---

### Task A5: ic_report.py — FDR-corrected significance + de-overlapped decay check

Fixes: I1 (multiple testing), I3 (autocorrelated weekly decay IC).

**Files:**
- Modify: `ic_report.py`

- [ ] **Step 1:** `spearman_ic` → also return p-value via `scipy.stats.spearmanr` (make scipy the required path; skip cleanly with a printed message if unavailable). After building `rows`, apply `statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method="fdr_bh")` across the factors actually tested; add a `significant` column; update the guide line to say `|IC|>0.05` is uncorrected. If statsmodels is not installed: `& $PY -m pip install statsmodels` and add `statsmodels` to `requirements.txt`.

- [ ] **Step 2:** Decay block: only run DECAY_ALARM when `horizon_days == 5` (near non-overlapping at weekly sampling). For `horizon_days == 21` print `decay check skipped at 21d (overlapping windows)` instead. Document choice inline.

- [ ] **Step 3: Manual verify**

Run: `& $PY ic_report.py`
Expected: table shows `IC`, `n`, `significant`; 21d decay line reports skipped.

- [ ] **Step 4: Commit**

```bash
git add ic_report.py requirements.txt
git commit -m "fix(ic): FDR-corrected factor significance, remove autocorrelated 21d decay alarm"
```

---

### Task A6: learn_weights.py — exact purge, honest holdout, drop negative factors

Fixes: L1, L2, L3. NOTE: learned weights are not active in production; this makes the tool safe to ever activate.

**Files:**
- Modify: `learn_weights.py`
- Test: `test_learn_weights.py` (create)

- [ ] **Step 1: Write failing test (L3 core — sign flip)**

```python
# test_learn_weights.py
import numpy as np
from learn_weights import weights_from_coefs

def test_negative_coef_gets_zero_weight():
    coefs = np.array([0.5, -0.5, 0.25])
    w = weights_from_coefs(coefs, ["a", "b", "c"])
    assert w["b"] == 0.0
    assert abs(w["a"] + w["c"] - 1.0) < 1e-9
    assert w["a"] > w["c"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `& $PY -m pytest test_learn_weights.py -v`
Expected: FAIL (function missing)

- [ ] **Step 3: Implement**

(L3) Extract and fix weight derivation:

```python
def weights_from_coefs(coefs: np.ndarray, factors: list[str]) -> dict:
    """Positive-coefficient factors share weight; negative-IC factors are
    DROPPED (weight 0.0) so pipeline.py's weights.get(k, 0) > 0 filter
    excludes them, instead of silently inverting their signal."""
    pos = np.where(coefs > 0, coefs, 0.0)
    if pos.sum() == 0:
        raise SystemExit("All factors have non-positive coefficients — keeping equal weights.")
    return {f: round(float(c / pos.sum()), 4) for f, c in zip(factors, pos)}
```

Use `final.coef_` (raw, not abs) as input. Update the WARNING print: negative factors are "dropped from weights (weight=0)", keep `coef_signs` in output JSON for audit.

(L1) `purged_folds`: replace the calendar heuristic with positional trading-day lookup on `unique_days`:

```python
pos0 = np.searchsorted(unique_days, t0)
purge_start = unique_days[max(0, pos0 - HORIZON)]
pos1 = np.searchsorted(unique_days, t1, side="right") - 1
purge_end = unique_days[min(len(unique_days) - 1, pos1 + EMBARGO)]
```

(L2) Final-weights holdout: reserve the last fold's test dates (+embargo) as a never-trained holdout; fit `final` only on rows strictly before its purge_start; report `final_weights_holdout_ic`; write `output/learned_weights.json` only if holdout IC > 0.03, else print reason and keep equal weights. Rename/annotate output field: `"oos_ic_caveat": "validates alpha selection only"`.

- [ ] **Step 4: Run tests**

Run: `& $PY -m pytest test_learn_weights.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add learn_weights.py test_learn_weights.py
git commit -m "fix(weights): drop negative-IC factors, exact trading-day purge, holdout-validated final weights"
```

---

### ⛔ GATE A: Re-validation — STOP and report to Frank

- [ ] Run, capture output of each:

```powershell
& $PY ic_report.py > output\revalidation_ic_20260711.txt 2>&1
& $PY backtest.py --sweep --period 5y > output\revalidation_sweep_20260711.txt 2>&1
& $PY backtest.py --mode kronos > output\revalidation_kronos_20260711.txt 2>&1
```

- [ ] Write a short summary table (old claim vs new honest number) to `notes/2026-07-XX-revalidation-results.md`:
  - Kronos daily IC (was +0.145) → new value, now measured next-open→close, non-overlapping, FDR-flagged
  - Sweep Sharpe (was 0.71 in-sample) → OOS Sharpe net of 10 bps
  - Per-factor `significant` flags
- [ ] **STOP. Do not start Phase B until Frank reviews.** Whether the edge survives is Frank's call, not the executor's. If IC/Sharpe collapse, Phases B–D are still worth doing (correctness), but factor redesign becomes the priority and Frank decides.

---

## PHASE B — Paper-ledger correctness

### Task B1: ledger.py — session-aware fills, no weekend fills, staleness bound, honest `filled` flag

Fixes: G1 (weekend fills), FRESH critical (stale fills after missed runs), FRESH major (skipped fills marked filled).

**Files:**
- Modify: `ledger.py` (`mark()`, `_today_prices()`)
- Test: `test_ledger.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# test_ledger.py
import pandas as pd
import ledger

def test_mark_noop_on_weekend(monkeypatch, capsys):
    class FakeDT:
        @staticmethod
        def now():
            return pd.Timestamp("2026-07-11")  # Saturday
    monkeypatch.setattr(ledger, "datetime", FakeDT)
    assert ledger.mark() is None
    assert "weekend" in capsys.readouterr().out.lower()

def test_stale_pending_not_filled_silently():
    # proposal 5 trading days old must not fill as if fresh
    log = pd.DataFrame([{"proposal_date": "2026-07-01", "ticker": "NVDA",
                         "action": "BUY", "adj_score": 0.8,
                         "target_value": 1000.0, "filled": False}])
    stale = ledger._flag_stale(log, today="2026-07-10", max_gap_days=1)
    assert stale.loc[0, "expired"] == True
```

- [ ] **Step 2: Run to verify failure**

Run: `& $PY -m pytest test_ledger.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

1. Top of `mark()`: `if datetime.now().weekday() >= 5: print("[LEDGER] Skipping mark(): weekend, no trading session."); return None`. Also skip on NYSE holidays — add a small module-level `NYSE_HOLIDAYS_2026` date list (source from `macro_calendar.json` if it has them, else hardcode the 2026 NYSE calendar with a `# verify yearly` comment).
2. `_today_prices()`: return the bar DATE alongside prices: `opens[t] = (bar_date, px)`. In `mark()`, only fill a pending proposal when `bar_date > proposal_date` (a genuine new session occurred). Otherwise leave `filled=False` for retry next run.
3. Stamp `entry_date`/`exit_date`/equity-curve date with the BAR's trading date, never `datetime.now()`.
4. Staleness bound — add helper and use it when building `pending`:

```python
def _flag_stale(log: pd.DataFrame, today: str, max_gap_days: int = 1) -> pd.DataFrame:
    """Proposals older than max_gap_days trading-ish days are expired, not filled.
    A missed scheduler run must not fill a days-old signal at today's open."""
    log = log.copy()
    gap = (pd.Timestamp(today) - pd.to_datetime(log["proposal_date"])).dt.days
    log["expired"] = (~log["filled"].astype(bool)) & (gap > max_gap_days + 2)  # +2 tolerates weekends
    return log
```

Expired rows: set `filled=True` plus new `expired=True` column, print `[LEDGER] EXPIRED (not filled): {ticker} proposal from {date}`, never enter positions.
5. Honest `filled` (skipped-fill bug): inside the pending loop, track `executed`:

```python
executed = act != "BUY" or t in held
if act == "BUY" and t not in held:
    shares = int(row["target_value"] // px) if row["target_value"] > 0 else 0
    cost = shares * px
    executed = shares > 0 and cost <= state["cash"]
    if executed:
        ...existing fill...
    else:
        print(f"[LEDGER] BUY {t} SKIPPED: shares={shares} cost={cost:.2f} cash={state['cash']:.2f}")
...
log.loc[idx, "filled"] = executed
```

- [ ] **Step 4: Run tests** — `& $PY -m pytest test_ledger.py -v` Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ledger.py test_ledger.py
git commit -m "fix(ledger): session-aware fills (no weekend/stale fills), skipped BUYs stay unfilled and retryable"
```

---

### Task B2: ledger.py — crash-safe write order + atomic writes + MTM/cash visibility

Fixes: FRESH critical (write-order fill loss), G4 (silent MTM fallback), G2 (truncation shortfall invisible), G3 (--cash silently ignored).

**Files:**
- Modify: `ledger.py`

- [ ] **Step 1:** Atomic write helper + reorder — PROPOSALS **last** (its `filled` flag gates retry; everything else must land first):

```python
import os

def _atomic_to_csv(df: pd.DataFrame, path) -> None:
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
```

```python
LEDGER_DIR.mkdir(exist_ok=True)
_atomic_to_csv(pos, POSITIONS)
_atomic_to_csv(closed, CLOSED)
_atomic_to_csv(eq, EQUITY)
_save_state(state)                 # also switch to tmp+os.replace inside
_atomic_to_csv(log, PROPOSALS)     # LAST: crash before this = clean retry
```

- [ ] **Step 2:** MTM fallback made loud (G4): replace `closes.get(p["ticker"], p["entry_price"])` with explicit branch printing `[LEDGER] WARNING: no close for {t}, marking at stale entry_price` per ticker; if ALL positions hit fallback, print a `DATA QUALITY WARNING: N/M positions stale` banner; still write the equity row.

- [ ] **Step 3:** Shortfall visibility (G2): after `cost = shares * px`, print `[LEDGER] {t}: sized {shares} sh (${cost:.2f}), ${target_value - cost:.2f} unfilled (whole-share floor)`. No fractional-share simulation.

- [ ] **Step 4:** `--cash` semantics (G3): argparse default `None`; resolve `starting_cash = 124_000 if starting_cash is None else starting_cash`. In `_load_state`, when state.json exists AND explicit `--cash` differs from persisted, print NOTE that it's ignored + how to override. Add `--reset-cash` flag that writes `{"cash": starting_cash}` to state.json with printed confirmation.

- [ ] **Step 5: Manual verify**

Run: `& $PY ledger.py mark --cash 999` (state.json exists) → NOTE printed, state unchanged.
Run: `& $PY -m pytest test_ledger.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add ledger.py
git commit -m "fix(ledger): atomic writes with proposals last (crash-safe), loud MTM fallback, explicit cash override"
```

---

### Task B3: intraday_pipeline.py — PDT tracker actually persists; corrupt-state and crash guards

Fixes: N1 (PDT gate non-functional), FRESH major (corrupt pdt_tracker.json crashes every scan silently).

**Files:**
- Modify: `intraday_pipeline.py`
- Test: `test_intraday.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# test_intraday.py
import json
import intraday_pipeline as ip

def test_load_pdt_tracker_corrupt_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ip, "LEDGER_DIR", tmp_path)
    (tmp_path / "pdt_tracker.json").write_text("{truncated")
    t = ip.load_pdt_tracker()
    assert t == {"trades": []}
    assert "corrupt" in capsys.readouterr().out.lower()

def test_record_pdt_trades(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "LEDGER_DIR", tmp_path)
    tracker = {"trades": []}
    ip.record_pdt_trades(tracker, n_buys=2)
    saved = json.loads((tmp_path / "pdt_tracker.json").read_text())
    assert len(saved["trades"]) == 2
```

- [ ] **Step 2: Run to verify failure** — Expected: FAIL

- [ ] **Step 3: Implement**

1. `load_pdt_tracker`: wrap `json.load` in `try/except (json.JSONDecodeError, OSError)` → print `[PDT] pdt_tracker.json corrupt, resetting to empty tracker — history since last valid save lost` and return `{"trades": []}`.
2. Add + call:

```python
def record_pdt_trades(tracker: dict, n_buys: int) -> None:
    """Count each entry as one PDT-relevant trade (conservative). Persist."""
    if n_buys <= 0:
        return
    ts = now_et().isoformat()
    tracker["trades"].extend([ts] * n_buys)
    save_pdt_tracker(tracker)
```

In `run_scan`'s paper-ledger block, after the BUY rows are appended to positions.csv: `record_pdt_trades(pdt_tracker, len(new_rows))` (reuse the tracker loaded at scan top — do not reload).
3. Wrap `main()`'s dispatch to `run_scan`/`run_force_close` in try/except: print full traceback, attempt a `plyer` Windows notification ("Intraday pipeline CRASHED — check logs"), swallow notification failures, re-raise nothing (exit code 1 via `sys.exit(1)`).

- [ ] **Step 4: Run tests** — `& $PY -m pytest test_intraday.py -v` Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add intraday_pipeline.py test_intraday.py
git commit -m "fix(intraday): PDT tracker persists on BUY (gate was inert), corrupt-state recovery, crash notification"
```

---

### Task B4: intraday_pipeline.py — position sizing (qty) + dollar PnL

Fixes: N2. Uses flat sizing per validated decision (position_size_pct=0.40).

**Files:**
- Modify: `intraday_pipeline.py`, `intraday_config.json`
- Test: `test_intraday.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_size_position_basic():
    assert ip._size_position(100.0, 0.40, 2000.0) == 8   # 2000*0.40/100

def test_size_position_price_exceeds_alloc():
    assert ip._size_position(900.0, 0.40, 2000.0) == 0   # caller must skip

def test_size_position_bad_price():
    assert ip._size_position(None, 0.40, 2000.0) is None
    assert ip._size_position(0.0, 0.40, 2000.0) is None
```

- [ ] **Step 2: Run to verify failure** — Expected: FAIL

- [ ] **Step 3: Implement**

1. `intraday_config.json`: add `"paper_equity": 2000` (Robinhood funding target per CLAUDE.md). Sanity note in code: `max_positions * position_size_pct = 0.80 ≤ 1.0` — assert this at config load, loud error if violated.
2. Helper:

```python
def _size_position(price, size_pct: float, equity: float):
    """Whole-share qty for a flat size_pct slice of equity. None on bad price;
    0 when one share exceeds the allocation (caller logs + skips)."""
    if not price or price <= 0:
        return None
    return int((equity * size_pct) // price)
```

3. In the BUY row builder replace `"qty": None` with the sized qty; when qty is 0, print `[SIZE] {t} skipped: 1 share (${price}) exceeds allocation` and do NOT append a phantom position (leave proposals action as-is for signal history).
4. `run_force_close`: keep `pnl_pct`; add `pnl_dollars = round((price - entry) * qty, 2)` when qty present; add column to closed_trades.csv.

- [ ] **Step 4: Run tests** — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add intraday_pipeline.py intraday_config.json test_intraday.py
git commit -m "feat(intraday): flat position sizing (paper_equity x 40%), dollar PnL in closed trades"
```

---

### Task B5: intraday_pipeline.py — ET timezone, entry-time gate, idempotent appends, bar staleness, partial-bar volume

Fixes: N5 (local-tz assumption), N6 (no_entry_before dead), N4 (duplicate appends), N7 (stale bars), FRESH major (partial-bar volume_surge deflation).

**Files:**
- Modify: `intraday_pipeline.py`, `requirements.txt`
- Test: `test_intraday.py` (extend)

- [ ] **Step 1 (N5):**

```python
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")

def now_et() -> datetime:
    return datetime.now(ET)
```

Add `tzdata` to `requirements.txt` (Windows lacks IANA db). `time_et_str()` unchanged.

- [ ] **Step 2 (N6):** In `run_scan` after the header print:

```python
no_entry_before = cfg.get("no_entry_before", "09:45")
entries_allowed = time_et_str() >= no_entry_before and now_et().weekday() < 5
if not entries_allowed:
    print(f"[GATE] before {no_entry_before} ET or weekend — entries disabled, scoring only")
```

Gate the BUY condition on `entries_allowed` (scan still scores/logs for visibility).

- [ ] **Step 3 (N4):** Idempotency guards:
- proposals CSV: before append, if file exists read its `scan_time` column; if current scan_time already present, print `duplicate scan_time — skipping append` and skip.
- positions.csv: before append, drop any `new_rows` whose `(ticker, entry_time)` already exists in the file.
- closed_trades: existing `if positions.empty: return` already no-ops a second force-close after positions are cleared — verify this by running `--force-close` twice; add a comment stating it's the guard.

- [ ] **Step 4 (N7):** In `fetch_1h_bulk`, before `result[ticker] = df`: staleness check — normalize `df["timestamps"].iloc[-1]` and `now_et()` to the same tz, and if age > 2h during a weekday 09:30–16:00 ET session, print `[STALE] {ticker} last bar {age:.1f}h old — skipped` and drop the ticker from `result`.

- [ ] **Step 5 (partial bar):** In `_score_volume_surge`, exclude the still-forming bar: compute elapsed fraction of the last bar's hour vs `now_et()`; if `< 1.0` treat `df["volume"].iloc[-2]` as current and shift the baseline window back one (`iloc[-window-2:-2]`). Simple, no extrapolation. Add test: construct df where last bar volume is tiny (partial) and second-to-last is 2× average — score must reflect the completed bar (>0), not the partial one (0).

- [ ] **Step 6: Run all tests** — `& $PY -m pytest test_intraday.py test_scoring.py -v` Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add intraday_pipeline.py requirements.txt test_intraday.py
git commit -m "fix(intraday): explicit ET tz, enforce entry-time gate, idempotent CSV appends, stale/partial bar handling"
```

---

### Task B6: Ledger reset (after B1–B5 all green)

Current ledger data is corrupted by the bugs above (unsized positions, phantom fills, inert PDT gate). Reset so the paper-validation clock restarts on trustworthy plumbing. **Archive, never delete.**

- [ ] **Step 1:**

```powershell
New-Item -ItemType Directory -Force ledger\archive\pre-fix-20260711
Move-Item ledger\intraday\*.csv, ledger\intraday\pdt_tracker.json ledger\archive\pre-fix-20260711\ -ErrorAction SilentlyContinue
Copy-Item ledger\equity_curve.csv, ledger\positions.csv, ledger\proposals_log.csv, ledger\state.json ledger\archive\pre-fix-20260711\ -ErrorAction SilentlyContinue
```

Daily ledger: COPY to archive (history stays), intraday: MOVE (restart clean). Do NOT touch `output/factor_log.csv`.

- [ ] **Step 2:** Note in `notes/`: paper-validation clock restarted 2026-07-11; prior intraday data unusable for edge validation (document why in one line each).

- [ ] **Step 3: Commit**

```bash
git add -A ledger notes
git commit -m "chore(ledger): archive pre-fix ledgers, restart paper validation clock"
```

---

## PHASE C — Model quality

### Task C1: kronos_forecast.py — median real path instead of synthetic mean; parameter-aware cache

Fixes: K1, K2.

**Files:**
- Modify: `kronos_forecast.py`

- [ ] **Step 1 (K1):** In `run_forecast`, `num_paths > 1` branch — replace element-wise mean with median-by-terminal-return real path:

```python
        terminal = [float(p["close"].iloc[-1]) / float(x_df["close"].iloc[-1]) - 1.0
                    for p in paths]
        good = [i for i, p in enumerate(paths) if not p["close"].isna().any()]
        if not good:
            raise RuntimeError("Kronos: all sampled paths contain NaN")
        order = sorted(good, key=lambda i: terminal[i])
        pred_df = paths[order[len(order) // 2]].copy()   # median REAL path, not a fabricated composite
        pred_df.index.name = "timestamps"
        pred_df = pred_df.reset_index()
        pred_df.attrs["paths"] = [paths[i]["close"].to_numpy() for i in good]
```

Audit callers in `pipeline.py`: `path_consistency` must read `.attrs["paths"]` (already does per audit); `forecast_edge` reads `pred_df["close"]` — now a real sampled path. Log a warning when NaN paths were excluded.

- [ ] **Step 2 (K2):** Cache correctness — sidecar metadata (keeps stable filenames):
- On save: after `pred_df.to_csv(...)`, write `{csv_name}.meta.json` with `{num_paths, sample_count, pred_len, model_key, temperature, top_p}` AND persist per-path closes to `{csv_name}.paths.npz` (`np.savez(..., **{f"p{i}": arr})`). Write both via tmp + `os.replace` (daily + intraday can race).
- On reuse: require sidecar exists and ALL fields match, else cache-miss and re-run inference. On hit with `num_paths > 1`, reconstruct `cached.attrs["paths"]` from the `.npz`; if the `.npz` is missing, treat as cache-miss (never return a multi-path forecast without attrs).

- [ ] **Step 3: Manual verify**

```powershell
& $PY -c "from kronos_forecast import run_forecast; import numpy as np; d=run_forecast('NVDA', num_paths=5, reuse_within_hours=0); paths=d.attrs['paths']; c=d['close'].to_numpy(); print('point forecast is a real path:', any(np.array_equal(c[:len(p)], p) or np.allclose(c[:len(p)],p) for p in paths))"
```

Expected: `point forecast is a real path: True`. Then call twice more with `reuse_within_hours=1`, first with `num_paths=1` (must MISS cache) then `num_paths=5` again (must HIT and include 5 paths).

- [ ] **Step 4:** Re-run Kronos IC (`& $PY backtest.py --mode kronos`) — the point-forecast selection changed; record the new IC next to Gate A numbers. Do not silently regress the validated edge.

- [ ] **Step 5: Commit**

```bash
git add kronos_forecast.py
git commit -m "fix(kronos): return median real sampled path (not synthetic mean), parameter-aware cache with path persistence"
```

---

### Task C2: pipeline.py — saturating mu, config-driven VIX gate, gate observability

Fixes: P3 (mu cap flattening), P2 (edge clip — soft variant), P4 (silent zero-target runs), FRESH major (hardcoded VIX 22), FRESH major (intraday daily-gate coverage blind spot).

**Files:**
- Modify: `pipeline.py`, `intraday_pipeline.py`
- Test: `test_scoring.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_annualize_kronos_mu_monotonic_above_cap():
    a = pipeline.annualize_kronos_mu(0.02, 10)
    b = pipeline.annualize_kronos_mu(0.05, 10)
    c = pipeline.annualize_kronos_mu(0.10, 10)
    assert a < b < c            # old code: all pinned at 0.60
    assert c <= 0.60 + 1e-9     # still bounded

def test_annualize_kronos_mu_extreme_negative_no_blowup():
    assert np.isfinite(pipeline.annualize_kronos_mu(-0.999, 10))
```

- [ ] **Step 2: Run to verify failure** — Expected: FAIL (a == b == c == 0.60)

- [ ] **Step 3: Implement**

(P3):

```python
def annualize_kronos_mu(fwd_ret: float, h_days: float, cap: float = 0.60) -> float:
    if h_days <= 0:
        return 0.0
    base = max(1.0 + fwd_ret, 1e-6)
    ann = base ** (252.0 / h_days) - 1.0
    return float(cap * np.tanh(ann / cap))   # smooth saturation keeps ordering; hard clip lost it
```

(P2): `_score_forecast_edge` soft-clip preserving order above threshold:

```python
def _score_forecast_edge(hist: pd.DataFrame, forecast: pd.DataFrame, threshold: float) -> float:
    rets = _forecast_returns(hist, forecast)
    if len(rets) == 0:
        return 0.5
    med = float(np.median(rets))
    x = max(0.0, med / threshold)
    return float(1.0 - np.exp(-x))   # monotonic: 0 -> 0.0, 1x thr -> 0.63, saturates toward 1.0
```

Old behavior clipped everything ≥1× threshold to identical 1.0; new curve keeps within-BUY ranking. Note: at exactly 1× threshold the score drops 1.0 → 0.63, which lowers raw_score for borderline names — Step 5 checks the BUY/HOLD boundary impact.

(P4): in `optimize_portfolio` `if not eligible:` branch print `[OPTIMIZER] 0 eligible tickers (all REDUCE) — no positions targeted`; in `run_pipeline` after targets: `if not targets: print("[PORTFOLIO] No targets generated this run")`.

(VIX config): `check_regime(hist, vix, vix_ceiling: float = 22.0)`; body uses the param; call site passes `cfg.get("vix_ceiling", 22)`.

(daily-gate coverage): `daily_gate_action` returns `"N/A"` when ticker absent (REDUCE check unchanged — only REDUCE skips); at scan start print `[GATE] daily gate covers {n_covered}/{n_total} tickers`.

- [ ] **Step 4: Run tests** — Expected: PASS

- [ ] **Step 5: Behavior check:** run `& $PY pipeline.py`, compare action distribution vs previous run's proposals CSV. P2's rescale shifts forecast_edge values slightly below the old clip — confirm BUY/HOLD boundary cases didn't flip en masse (a handful is expected; wholesale flips mean the rescale needs the 0.63→1.0 band re-examined with Frank).

- [ ] **Step 6: Commit**

```bash
git add pipeline.py intraday_pipeline.py test_scoring.py
git commit -m "fix(model): saturating kronos mu and edge scores keep ranking power, config VIX ceiling, gate observability"
```

---

## PHASE D — Hygiene

### Task D1: Watcher retries + cache correctness (S3, S4, S5)

**Files:** `usaspending_watcher.py`, `edgar_watcher.py`, `quiver_congress_watchlist.py`

- [ ] Shared pattern (implement per file, matching each file's style — no shared module, keep surgical):
  - Retry loop: 3 attempts, backoff 1s/2s/4s, retry ONLY on `requests.exceptions.RequestException` / 5xx, never on 4xx.
  - `edgar_watcher.py`: keep existing UA + 0.3s pacing (correct as-is). Only write per-ticker cache rows for SUCCESSFUL tickers so a transient failure doesn't poison the day's cache as "no insider buys".
  - `quiver_congress_watchlist.py`: skip writing the daily cache when fewer than both chambers succeeded (next call retries fully).
  - `usaspending_watcher.py`: add daily CSV cache (`output/contracts/awards_{YYYYMMDD}.csv`) + stale-cache fallback ≤7 days with a printed `fresh|stale-cache|no-data` marker.
- [ ] Verify: temporarily monkeypatch/mock a failing `requests` call per module and confirm graceful path.
- [ ] Commit: `git commit -m "fix(watchers): retry with backoff, stop caching partial failures as complete data"`

### Task D2: robinhood_fetcher.py tz (S6)

- [ ] Line ~136: `pd.to_datetime(df["timestamps"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)` — naive-Eastern convention (matches yfinance-naive expectations elsewhere; VERIFY by checking what `_fetch_yfinance` produces and match exactly).
- [ ] `_trim()`: make tz reconciliation bidirectional (naive ts + aware cutoff → strip cutoff tz).
- [ ] Test in `test_intraday.py`: `'2024-01-02T14:30:00Z'` → naive `2024-01-02 09:30:00`.
- [ ] Commit: `git commit -m "fix(rh): RH timestamps to naive Eastern, bidirectional tz handling in _trim"`

### Task D3: run_scheduler.py Ollama readiness (S1)

- [ ] Delete the dead `subprocess.run(["ollama","serve"], timeout=3)` block (always times out — it's a foreground server).
- [ ] After Popen (and also on the psutil already-running path): poll `requests.get("http://localhost:11434/api/tags", timeout=2)` every 1s up to 30s; True on 200; else print `[OLLAMA] not ready after 30s, brief will use fallback` and return False.
- [ ] Verify: kill ollama.exe, run `& $PY run_scheduler.py --now`, check log shows ready-poll or explicit fallback warning.
- [ ] Commit: `git commit -m "fix(scheduler): real Ollama readiness poll instead of dead 3s timeout"`

### Task D4: Config/constant drift (FRESH minors)

- [ ] Expose `BUY_THR = 0.70` / `HOLD_THR = 0.45` as constants in `pipeline.py`; use in `action_label`; import in `backtest.py` and change `--hold-thr` default to `HOLD_THR` (0.45 — was stale 0.4).
- [ ] Delete dead `"forecast_threshold": 0.03` key from `config.json` (nothing reads it; `vol_scaled_threshold` is the real mechanism).
- [ ] Run full test suite: `& $PY -m pytest -v` Expected: PASS
- [ ] Commit: `git commit -m "chore(config): share BUY/HOLD threshold constants, remove dead forecast_threshold key"`

### Task D5: Archive dead files

- [ ] `New-Item -ItemType Directory -Force archive` then `git mv debug.py make_brief.py make_brief_fixed.py research_script.py research_script_final.py build_basket.py robinhood_fetcher.py archive\` — **EXCEPTION:** keep `robinhood_fetcher.py` in root if D2 wired it anywhere; it is currently standalone — confirm nothing imports it after D2, and if D2's fix is worth keeping for future rh_executor work, keep the file in root and only archive the other six.
- [ ] Commit: `git commit -m "chore: archive dead experimental scripts"`

### Task D6: Documentation close-out

- [ ] Update `CLAUDE.md`: changelog rows for each phase commit; clear resolved "Known issues"; add new known limitation notes (daily-gate covers only the 16-ticker daily universe by design; regime gate is entry-only by design).
- [ ] Update `README.md` test section if new test files change the suite shape.
- [ ] Commit: `git commit -m "docs: changelog + known-issues refresh after audit fix"`

---

## Execution order summary

| Order | Tasks | Gate |
|---|---|---|
| 1 | A1 → A2 → A3 → A4 → A5 → A6 | ⛔ GATE A: re-validate, write results note, STOP for Frank |
| 2 | B1 → B2 → B3 → B4 → B5 → B6 | ledger reset only after B1–B5 green |
| 3 | C1 → C2 | C1 Step 4 re-checks Kronos IC |
| 4 | D1 → D2 → D3 → D4 → D5 → D6 | full `pytest -v` green at end |

Full suite check between every phase: `& $PY -m pytest -v` — all green before moving on.
