# Phase E — Alpha Upgrades Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the pipeline from "one absolute score + bolt-on modifiers" to a measured multi-layer system: cross-sectional ranking, forecast-uncertainty gating, measured alt-data signals, regime-aware weights, edge-scaled sizing, and codified kill criteria.

**Architecture:** Instrument-first. Every new layer follows the same lifecycle: (1) LOG the signal without letting it trade, (2) VALIDATE with the honest IC harness once enough history accumulates, (3) ACTIVATE only on FDR-significant evidence. Nothing new trades untested — that discipline IS the edge. Buildable-now tasks (E1–E4, E7, E8) ship immediately; activation of data-gated layers (E5, E6, meta-model) is threshold-triggered, not calendar-triggered.

**Tech Stack:** Python 3.14 (`C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe`, alias `$PY`), pandas, numpy, scipy/statsmodels, existing pytest suite.

---

## Prerequisites (hard)

1. **Audit plan complete through Phase C** (`2026-07-11-audit-fix-plan.md`) — especially:
   - Gate A passed (honest IC baseline exists)
   - Task C1 (Kronos median real path + `.npz` path persistence — E2 is impossible while cache hits lose `attrs["paths"]`)
   - Task A3 (vol_context fixed — factor values trustworthy)
2. Same executor ground rules as the audit plan (interpreter, commits per task, factor_log history is append-only).
3. **Scout-verified integration facts** (2026-07-11, verify still true before each task):
   - `factor_log.csv`: 28 cols; `adj_score` is a DUPLICATE of `score` (both set to raw_score at pipeline.py:655-658) — free to repurpose
   - `compute_score(hist, forecast, ticker, contract_signals, threshold, weights)` returns dict; congress/insider mods applied by CALLER at pipeline.py:576-588 via `np.clip(raw + cong_mod + ins_mod, 0, 1)`
   - `_forecast_returns()` (pipeline.py:235-241) returns per-path terminal returns list — dispersion input already exists
   - `ic_report.py` FACTORS (line 27) excludes `congress_mod`/`insider_mod` though both are logged per row
   - `config.json`: `congress_enabled: false`, `insider_enabled: true`, `num_paths: 3`, `factor_weights: "equal"`
   - Daily runtime ~4–7 min for 14 tickers (16 minus SPY/QQQ), per-ticker sequential, ~16–30 s/ticker

---

## ⚠️ Decisions Frank must approve BEFORE dispatch (executor: do not guess)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Insider modifier: currently ACTIVE (+0.10 on a [0,1] score — flips HOLD→BUY at 0.60) but has NEVER been IC-measured. Switch to log-only until validated? | **Yes, log-only.** Unmeasured +0.10 is the biggest unvalidated lever in the system. |
| D2 | Universe expansion 16 → ~50 tickers (E3). Mechanical rule: current 16 + all non-leveraged single names & sector ETFs from intraday_config.json (drop TQQQ/SQQQ/SPXL/UVXY). Runtime → ~15–25 min daily. | **Yes.** Ranks over 14 names are statistically weak; 45+ makes cross-sectional signal real. Batch fetch (E3 Step 1) claws back several minutes. |
| D3 | `num_paths` 3 → 7 (E2). Adds ~4 Kronos inferences/ticker ≈ minutes of runtime across the universe. Dispersion over 3 paths is nearly meaningless; 7 is the useful minimum. | **Yes** — pairs with D2; total run stays under ~35 min, fine for a 4:30pm batch. |

Record approvals in this file under this table before starting E2/E3/E4.

**APPROVED by Frank 2026-07-11: D1 yes (insider → log-only), D2 yes (universe → ~60), D3 yes (num_paths → 7).**

**EXECUTOR NOTE (2026-07-19, pre-E3):** `intraday_config.json` has grown since D2 was estimated. Applying the mechanical rule literally (current 16 + all non-leveraged singles/ETFs from `intraday_config.json`, drop TQQQ/SQQQ/SPXL/UVXY) now yields **~80 tickers**, not ~60. This inflates daily runtime beyond D2's ~15–25 min estimate, compounding with D3's num_paths 3→7 (more Kronos inferences/ticker). Executor is applying the rule literally per plan text (not re-deciding the approved count) — this is Frank's call on whether to prune further after seeing the actual runtime. See E3 section below and the executor's final report for the live-run timing.

---

## Task E1: Instrumentation — log forecast dispersion + missing signal values

Everything later phases need must start accumulating history NOW.

**Files:**
- Modify: `pipeline.py` (`compute_score`, rows dict), `intraday_pipeline.py`, `kronos_forecast.py` (nothing — C1 already persists paths)
- Test: `test_scoring.py` (extend)

- [ ] **Step 1: Write failing tests**

```python
def test_path_dispersion_known_values():
    # terminal returns [-0.02, 0.00, 0.02] -> std ~0.0163, median 0.0
    d = pipeline._path_dispersion([-0.02, 0.0, 0.02])
    assert abs(d - np.std([-0.02, 0.0, 0.02])) < 1e-9

def test_path_dispersion_single_path():
    assert pipeline._path_dispersion([0.01]) is None   # undefined for 1 path
```

- [ ] **Step 2: Run to verify failure** — `& $PY -m pytest test_scoring.py -v` → FAIL

- [ ] **Step 3: Implement**

`pipeline.py`:

```python
def _path_dispersion(rets: list[float]) -> float | None:
    """Std of per-path terminal returns. Model uncertainty: wide = low conviction.
    None when fewer than 2 paths (dispersion undefined)."""
    if len(rets) < 2:
        return None
    return float(np.std(rets))
```

In `compute_score`, after `kronos_fwd_ret` is computed from `rets = _forecast_returns(hist, forecast)`: add to the returned dict `"path_dispersion": _path_dispersion(rets)` and `"n_paths": len(rets)`. In the caller's rows dict (~line 655-683) add both columns so they land in factor_log.csv. (New columns appended at end — the audit plan's A2 rewrite handles schema migration via full read-merge-write.)

`intraday_pipeline.py`: add `"news_sent": n.get("sent", 0.0)` to BOTH row branches (skip and non-skip) of the scan loop — currently the numeric sentiment is dropped intraday and only the flag survives; meta-labeling needs the number.

- [ ] **Step 4: Run tests + one live run**

`& $PY -m pytest -v` → PASS. Then `& $PY pipeline.py` → confirm factor_log gains `path_dispersion`, `n_paths` columns with finite values for forecast tickers.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py intraday_pipeline.py test_scoring.py
git commit -m "feat(instrument): log kronos path dispersion + n_paths daily, news_sent intraday"
```

---

## Task E2: Kronos dispersion gate (log → threshold → gate)

Model uncertainty as a tradeable signal: when sampled paths disagree wildly, the point forecast is noise — skip.

**Requires D3 approved.** Depends on audit Task C1 (paths survive cache via .npz).

**Files:**
- Modify: `config.json`, `pipeline.py`
- Test: `test_scoring.py` (extend)

- [ ] **Step 1:** `config.json`: `"num_paths": 7`, add `"dispersion_gate": "log"` (modes: `"log"` | `"active"`), add `"dispersion_max_rel": 3.0` (placeholder — tuned in Step 4).

- [ ] **Step 2: Failing test**

```python
def test_dispersion_gate_blocks_wide_paths():
    # relative dispersion = std/|median| ; gate blocks when > max_rel
    assert pipeline.dispersion_ok(0.01, 0.002, max_rel=3.0) is True    # 0.2 < 3
    assert pipeline.dispersion_ok(0.001, 0.02, max_rel=3.0) is False   # 20 > 3
    assert pipeline.dispersion_ok(0.0, 0.02, max_rel=3.0) is False     # zero median, wide paths
    assert pipeline.dispersion_ok(0.01, None, max_rel=3.0) is True     # no dispersion info -> don't block
```

- [ ] **Step 3: Implement**

```python
def dispersion_ok(median_ret: float, dispersion: float | None, max_rel: float = 3.0) -> bool:
    """False when path std overwhelms the median signal (|median| acts as the
    signal scale; floor avoids div-by-zero). None dispersion = no info = pass."""
    if dispersion is None:
        return True
    scale = max(abs(median_ret), 1e-4)
    return dispersion / scale <= max_rel
```

Wire into the action decision next to the news veto (~line 653): when `cfg["dispersion_gate"] == "active"` and `act == "BUY"` and not `dispersion_ok(...)` → `act = "HOLD"`, log `regime_note`-style reason `disp={ratio:.1f}`. In `"log"` mode compute and print, never alter action. **Ship in `"log"` mode.**

- [ ] **Step 4 (after ≥10 run_dates of dispersion history):** tune threshold from data, not vibes:

```powershell
& $PY - <<'EOF'
import pandas as pd, numpy as np
log = pd.read_csv("output/factor_log.csv")
d = log.dropna(subset=["path_dispersion", "kronos_fwd_ret"])
rel = d["path_dispersion"] / d["kronos_fwd_ret"].abs().clip(lower=1e-4)
print(rel.describe(percentiles=[.5, .75, .9]))
EOF
```

Set `dispersion_max_rel` at the ~75th percentile (gates the worst quartile of conviction). Then use the fixed IC harness: compare IC of `kronos_fwd_ret` on rows BELOW vs ABOVE the threshold. If below-threshold IC is materially higher → flip `dispersion_gate` to `"active"`. If no separation → leave in log mode, note it, move on (negative results are results).

- [ ] **Step 5: Commit** — `git commit -m "feat(kronos): path-dispersion conviction gate (log mode), num_paths 7"`

---

## Task E3: Cross-sectional rank scoring + universe expansion

Absolute prediction is the hard problem; relative ranking is the tractable one. `adj_score` is a duplicate column today — it becomes the rank score.

**Requires D2 approved.**

**Files:**
- Modify: `pipeline.py`, `config.json`
- Test: `test_scoring.py` (extend)

- [ ] **Step 1: Batch fetch first (runtime guard).** Replace per-ticker `fetch_history` loop calls with one bulk download:

```python
def fetch_history_bulk(tickers: list[str], period: str = "3y") -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, period=period, interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    out = {}
    for t in tickers:
        try:
            df = raw[t].dropna().rename(columns=str.lower)
            if len(df) >= 60:
                out[t] = df
        except KeyError:
            print(f"[FETCH] {t}: no data")
    return out
```

Call once in `run_pipeline()`; the per-ticker loop consumes from the dict. Add per-ticker Kronos timing print (`[TIMING] {ticker}: {sec:.1f}s`) so future universe decisions have data.

- [ ] **Step 2: Universe.** `config.json` tickers → current 16 + non-leveraged names from intraday_config.json (drop TQQQ/SQQQ/SPXL/UVXY; drop dupes). ~60 names. SPY/QQQ stay benchmark-only (excluded from forecast_tickers — existing behavior, now documented).

- [ ] **Step 3: Failing test**

```python
def test_rank_scores_are_percentiles():
    raws = {"A": 0.80, "B": 0.60, "C": 0.40, "D": 0.20}
    ranks = pipeline.rank_scores(raws)
    assert ranks["A"] == 1.0 and ranks["D"] == 0.25
    assert ranks["B"] == 0.75 and ranks["C"] == 0.5

def test_rank_scores_single_ticker_neutral():
    assert pipeline.rank_scores({"A": 0.9}) == {"A": 0.5}
```

- [ ] **Step 4: Implement**

```python
def rank_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Cross-sectional percentile rank (1.0 = best of today's universe).
    Relative ranking is a weaker claim than absolute prediction — and a
    more defensible one."""
    if len(raw_scores) < 2:
        return {t: 0.5 for t in raw_scores}
    s = pd.Series(raw_scores)
    return s.rank(pct=True).round(4).to_dict()
```

In `run_pipeline()` after all tickers scored: `ranks = rank_scores({t: s["raw_score"] for t, s in scores.items()})`; set the row's `adj_score` to `ranks[ticker]` (breaking the duplicate — `score` stays raw). Action rule: **unchanged BUY logic on raw_score** (absolute floor stays, validated) PLUS new requirement `adj_score >= cfg.get("min_rank", 0.75)` for BUY — configurable, ship at 0.75 (top quartile). `ic_report.py` already includes `adj_score` in FACTORS — after this change it measures the rank signal's IC vs `score`'s raw IC head-to-head. That comparison decides which becomes primary in 2–3 months.

- [ ] **Step 5:** Full suite + one live run; confirm proposals CSV shows distinct score vs adj_score, BUY count sane (not zero, not everything).

- [ ] **Step 6: Commit** — `git commit -m "feat(scoring): cross-sectional rank as adj_score, top-quartile BUY filter, 60-name universe, bulk fetch"`

---

## Task E4: Alt-data measurement — stop flying blind on modifiers

congress/insider mods move the score up to ±0.10 yet are invisible to every IC/weights tool. Measure before trusting.

**Requires D1 decision.**

**Files:**
- Modify: `pipeline.py`, `config.json`, `ic_report.py`

- [ ] **Step 1:** Replace boolean enables with modes in `config.json`:
  - `"congress_mode": "log"` (was `congress_enabled: false` — off gathered zero evidence; log gathers evidence at zero trading risk)
  - `"insider_mode": "log"` (per D1 — was silently active)
  - Keep old keys working: `mode = cfg.get("congress_mode") or ("active" if cfg.get("congress_enabled") else "off")`.

- [ ] **Step 2:** `pipeline.py` caller block (~576-588): always FETCH and COMPUTE `cong_mod`/`ins_mod` when mode != "off"; only ADD to raw_score when mode == "active":

```python
if congress_mode != "off":  # fetch + compute cong_mod as today
    ...
applied_cong = cong_mod if congress_mode == "active" else 0.0
applied_ins  = ins_mod  if insider_mode  == "active" else 0.0
s["raw_score"] = round(float(np.clip(s["raw_score"] + applied_cong + applied_ins, 0.0, 1.0)), 3)
```

Log the COMPUTED values (`congress_mod`, `insider_mod` columns — already exist) regardless of application, plus new column `mods_applied` ("both"/"congress"/"insider"/"none") so history is interpretable.

- [ ] **Step 3:** `ic_report.py` FACTORS: append `"congress_mod", "insider_mod"`. They now get IC + FDR significance like every other signal.

- [ ] **Step 4:** Promotion rule (document in this file + CLAUDE.md): a modifier goes `log → active` only when FDR-significant IC over ≥60 run_dates with |IC| ≥ 0.03. Demotion: significance lost on 2 consecutive monthly checks → back to log.

- [ ] **Step 5: Commit** — `git commit -m "feat(altdata): congress/insider to measured log-mode with promotion criteria, IC coverage added"`

---

## Task E5: Regime-conditional weights (infrastructure now, activation data-gated)

Factors behave differently calm vs stressed. `vix` + `regime_ok` already logged per row — the split is free.

**Files:**
- Modify: `ic_report.py`, `learn_weights.py`

- [ ] **Step 1:** `ic_report.py`: add a per-regime section — split rows by `vix < 20` (calm) vs `>= 20` (stressed), print the factor IC table per bucket with per-bucket `n` run_dates. Purely diagnostic.

- [ ] **Step 2:** `learn_weights.py`: add `--regime {calm,stressed,all}` flag filtering rows before fitting; output file becomes `learned_weights_{regime}.json`. **Hard data gate in code:** refuse to fit any bucket with < 60 distinct run_dates:

```python
n_days = df["run_date"].nunique()
if n_days < 60:
    raise SystemExit(f"Regime '{args.regime}': only {n_days} run_dates (<60) — refusing to fit noise.")
```

- [ ] **Step 3:** `pipeline.py` future hook (do NOT activate): `factor_weights: "regime"` config value documented as reserved; activation is a separate future task after both buckets clear the gate AND holdout IC (audit Task A6) passes per bucket. Today's data: 20 run_dates total — months away by design.

- [ ] **Step 4: Commit** — `git commit -m "feat(regime): per-regime IC diagnostics + gated regime weight fitting"`

---

## Task E6: Edge-scaled vol targeting (fractional-Kelly direction, data-gated)

Sizing converts edge into compounding. Scale gross exposure by MEASURED trailing performance of the honest ledger — more when the edge is real, less when it isn't.

**Files:**
- Modify: `pipeline.py`, `config.json`
- Test: `test_scoring.py` (extend)

- [ ] **Step 1: Failing test**

```python
def test_edge_vol_scale():
    # trailing sharpe 1.0, fraction 0.5 -> scale 1.0 + 0.5*(1.0-0.7)=1.15, clamped [0.5, 1.3]
    assert abs(pipeline.edge_vol_scale(1.0, base_sharpe=0.7, fraction=0.5) - 1.15) < 1e-9
    assert pipeline.edge_vol_scale(-1.0, 0.7, 0.5) == 0.5     # losing streak -> floor
    assert pipeline.edge_vol_scale(None, 0.7, 0.5) == 1.0     # no history -> neutral
```

- [ ] **Step 2: Implement**

```python
def edge_vol_scale(trailing_sharpe: float | None, base_sharpe: float = 0.7,
                   fraction: float = 0.5, lo: float = 0.5, hi: float = 1.3) -> float:
    """Fractional-Kelly-flavored gross scaling: risk up modestly when realized
    edge beats target, cut hard when it doesn't. Neutral without history."""
    if trailing_sharpe is None:
        return 1.0
    return float(np.clip(1.0 + fraction * (trailing_sharpe - base_sharpe), lo, hi))
```

Trailing Sharpe from `ledger/equity_curve.csv`: last 60 rows' daily equity pct-change, annualized mean/std; `None` when < 60 rows. Apply as a multiplier on `target_vol` BEFORE `portfolio_vol_scalar` (i.e., effective target vol = `cfg["target_vol"] * edge_vol_scale(...)`), so the existing clamps (`vol_scalar_min/max`) still bound the final result — layered safety.

- [ ] **Step 3:** Config: `"kelly_fraction": 0.0` — **ships OFF** (0.0 → scale is 1.0 regardless). Activation criterion: ≥60 trading days of post-reset honest equity curve (audit Task B6 restarted the clock). Frank flips to 0.5 after reviewing.

- [ ] **Step 4: Commit** — `git commit -m "feat(sizing): edge-scaled vol targeting, shipped dormant behind kelly_fraction=0"`

---

## Task E7: Meta-label dataset extractor (harness now, model when data ready)

Meta-labeling: a second model that predicts WHEN the primary signal pays, from context (VIX, regime, dispersion, news, earnings distance). ~313 rows today — training now = overfit garbage. Build the dataset pipe; train at ≥1000 rows (~4 months).

**Files:**
- Create: `build_meta_dataset.py`
- Test: `test_meta.py` (create)

- [ ] **Step 1: Failing test**

```python
# test_meta.py
import pandas as pd
from build_meta_dataset import label_rows

def test_label_rows_joins_outcomes():
    log = pd.DataFrame([
        {"run_date": "2026-07-01", "ticker": "NVDA", "action": "BUY", "vix": 18.0},
        {"run_date": "2026-07-01", "ticker": "CACI", "action": "HOLD", "vix": 18.0},
    ])
    fwd = pd.Series([0.03, -0.01], index=log.index)   # honest fwd returns (ic_report anchor)
    out = label_rows(log, fwd)
    assert list(out["label"]) == [1, 0]   # BUY row profitable -> 1; non-BUY excluded or 0 per spec
```

- [ ] **Step 2: Implement** `build_meta_dataset.py`:
- Reuse `ic_report.load_with_forward_returns()` (post-audit version — next-open anchor) for honest labels.
- Features per row (all already logged after E1/E4): `vix, regime_ok, path_dispersion, n_paths, news_sent, days_to_earnings, macro_event, forecast_edge, vol_context, adj_score(rank), congress_mod, insider_mod`.
- Label: `1` if the honest forward return of an actionable row (action == BUY, or raw_score ≥ 0.7 pre-gates) exceeds costs (10 bps), else `0`.
- Output: `output/meta_dataset.csv` + printed row count and readiness verdict: `READY (n>=1000)` / `NOT READY (n=… , need ~{k} more run_dates)`.
- **No model training in this task.** Training spec (logistic regression or gradient boosting, purged CV per audit A6 pattern, FDR-checked lift) is a future task unlocked by the READY verdict.

- [ ] **Step 3: Commit** — `git commit -m "feat(meta): meta-label dataset extractor with readiness gate"`

---

## Task E8: Codified kill/keep criteria — weekly health check

Discipline codified beats discipline remembered.

**Files:**
- Create: `health_check.py`
- Modify: Task Scheduler (manual step for Frank) or run_scheduler.py Friday hook

- [ ] **Step 1: Implement** `health_check.py` — read-only report, prints verdict per rule:

| Signal | Rule | Verdict output |
|---|---|---|
| Each factor | FDR-significant IC ≤ 0 two consecutive monthly checks | `KILL-CANDIDATE (needs Frank)` |
| Intraday system | after ≥10 post-fix trading days: score-vs-next-scan rank IC < 0.02 OR net pnl_dollars < 0 over ≥20 closed trades | `SHUT OFF SCHEDULE (needs Frank)` |
| Kronos daily | existing decay alarm (5d horizon, post-audit) | passthrough |
| congress/insider (log mode) | ≥60 run_dates AND FDR-significant AND \|IC\| ≥ 0.03 | `PROMOTE-CANDIDATE` |
| dispersion gate (log mode) | below-threshold IC − above-threshold IC ≥ 0.03 | `ACTIVATE-CANDIDATE` |
| Meta dataset | n ≥ 1000 | `TRAIN-READY` |
| Regime buckets | each ≥ 60 run_dates | `FIT-READY` |

Implementation: import from `ic_report` (post-audit: p-values + FDR available), read factor_log + intraday proposals + closed_trades. Every verdict prints evidence (IC, n, dates). **The script never acts — every kill/activate is a printed recommendation for Frank.** Persist each run to `output/health/health_{YYYYMMDD}.txt`.

- [ ] **Step 2:** Wire to Friday 4pm weekly audit ritual (CLAUDE.md already has that slot): add to run_scheduler or Task Scheduler — Frank's call, note in output.

- [ ] **Step 3: Commit** — `git commit -m "feat(health): weekly codified kill/promote criteria report"`

---

## Lifecycle timeline (what activates when)

| When | Event |
|---|---|
| Now (dispatch) | E1–E4, E7, E8 built. Rank scoring live (top-quartile BUY filter), dispersion + mods logging, health check running |
| +2 weeks (~10 run_dates) | E2 Step 4: tune dispersion threshold from logged data; intraday kill rule has enough trades to evaluate |
| +2–3 months (~60 run_dates) | congress/insider promotion decisions; regime-bucket fitting unlocks; kelly_fraction activation reviewable; score-vs-adj_score (raw vs rank) IC verdict |
| +4 months (~1000 meta rows) | Meta-model training task unlocked |

## Execution order

E1 → E4 → E3 → E2 → E7 → E5 → E6 → E8. Full `& $PY -m pytest -v` green between tasks. Every task's new signal must appear in the NEXT day's factor_log before the following task starts (one live run between tasks is the integration test).

## Hard boundaries for the executor

- Ship every new trading influence OFF or in log mode. The only behavior change that trades on day one is E3's top-quartile BUY filter (conservative: it only REMOVES trades, never adds).
- No threshold tuned on the same data that reports it (audit Gate A doctrine applies to every number here).
- All promote/activate/kill decisions print as recommendations — Frank decides, the system never self-activates a signal.
