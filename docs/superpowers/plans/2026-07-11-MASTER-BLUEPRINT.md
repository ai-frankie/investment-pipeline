# Investment Pipeline — Master Engineering Blueprint

**Role of this document:** the single map. Any future session (Claude or human) reads this first, sees system end-state, current position, and exactly how to dispatch the next executor. Detailed task-level plans live in their own files; this doc never duplicates their content, only points.

**Architect:** Fable (this doc + all phase plans). **Executors:** Sonnet, one per phase, dispatched via briefs in §5. **Verification:** scoped pytest per plan + Gate reviews by Frank. Writer ≠ executor ≠ verifier, per global routing rules.

---

## 1. End-state architecture (target system)

```
                        ┌─ DATA LAYER ─────────────────────────────┐
  yfinance (bulk 1d/1h) │  FRED macro   USASpending   EDGAR Form4  │
  Robinhood bars (opt.) │  Quiver congress   FinBERT news          │
                        └───────────────┬──────────────────────────┘
                                        ▼
                        ┌─ SIGNAL LAYER ───────────────────────────┐
                        │ price factors (momentum/trend/vol/qual)  │
                        │ Kronos forecast (median real path,       │
                        │   dispersion = conviction)               │
                        │ alt-data signals (log→validate→activate) │
                        └───────────────┬──────────────────────────┘
                                        ▼
                        ┌─ SCORING LAYER ──────────────────────────┐
                        │ raw_score (absolute, weighted factors)   │
                        │ adj_score (cross-sectional rank, ~60     │
                        │   names)                                 │
                        │ meta-model P(signal pays | context)      │
                        │   [data-gated]                           │
                        └───────────────┬──────────────────────────┘
                                        ▼
                        ┌─ GATE STACK (order matters) ─────────────┐
                        │ regime → VIX → macro blackout → earnings │
                        │ → news veto → dispersion gate → rank     │
                        │ floor → (intraday: PDT, time, max pos)   │
                        └───────────────┬──────────────────────────┘
                                        ▼
                        ┌─ SIZING LAYER ───────────────────────────┐
                        │ EF max-sharpe (Ledoit-Wolf) + kronos mu  │
                        │ portfolio vol targeting × edge-scaled    │
                        │   multiplier (fractional-Kelly, gated)   │
                        └───────────────┬──────────────────────────┘
                                        ▼
              ┌─ EXECUTION ──────────────┴─────────────────────────┐
              │ DAILY: proposals CSV → Frank approves → Fidelity   │
              │   (manual forever — IRA)                           │
              │ INTRADAY: paper ledger → [Phase F] rh_executor     │
              │   with human-armed kill-switch caps                │
              └──────────────────────────┬─────────────────────────┘
                                         ▼
                        ┌─ MEASUREMENT LOOP (the moat) ────────────┐
                        │ factor_log (append-only, honest labels)  │
                        │ ic_report (FDR, next-open anchor)        │
                        │ backtest (net-of-cost, OOS only)         │
                        │ health_check (codified kill/promote)     │
                        │ equity curve (crash-safe ledger)         │
                        └──────────────────────────────────────────┘
```

**Design invariants (permanent, every future phase inherits):**
1. Every signal lives LOG → VALIDATE → ACTIVATE. Nothing trades unmeasured.
2. No number is tuned on the window that reports it (Gate A doctrine).
3. Every activation/kill is a printed recommendation; Frank flips the switch.
4. factor_log history is append-only. Measurement code changes; data never.
5. All state writes atomic (tmp + os.replace); retry-gating file written last.
6. Fidelity IRA is proposal-only forever. Automation applies only to the Robinhood agentic account, capped, with kill switch.

---

## 2. Roadmap and current position

| Phase | Content | Plan doc | Status (2026-07-11) | Gate |
|---|---|---|---|---|
| A | Measurement integrity (6 tasks) | audit-fix-plan | ✅ DONE, commits 39950e2..f22267d | ⚠️ Gate A initially Branch 1 (IC +0.191, OOS net Sharpe 0.53); **post-C1 re-check supersedes → Branch 2** (IC +0.061, hit 50%) — mean-of-paths artifact removed |
| B | Ledger correctness (6 tasks) | audit-fix-plan | ✅ DONE, commits e1e602c..842c706 + B6 archive (uncommittable, ledger/ gitignored) | tests green + ledger reset 2026-07-11 |
| C | Model quality (2 tasks) | audit-fix-plan | ✅ DONE, commits d1627a2, 85ca3a6 | post-C1 IC re-check done: +0.061 → Branch 2 posture |
| D | Hygiene (6 tasks) | audit-fix-plan | ✅ DONE, commits 8664705..feafb8c; +X1/X2 addendum db0b237, 43e5c41 (loud scan-fail, force-close no-wipe) | scoped suite 55 green, Haiku-verified 2026-07-17 |
| E | Alpha upgrades (8 tasks) | phase-e-alpha-upgrades | ✅ DONE 2026-07-19, commits 8266205..5712fad (E1,E4,E3,E2,E7,E5,E6,E8), 62 tests green, Branch 2 posture. Report: notes/2026-07-18-phase-e-executor-report.md (filename misdated). Universe 16→**80** (mechanical rule output — Frank review pending), num_paths 7, only live behavior change = E3 rank floor on BUYs | Frank reviews: 80-ticker universe + wiring health_check.py into Friday ritual |
| F | Live execution bridge (rh_executor) | §4.1 (design here; task plan TBD) | 🧭 designed only | ≥10 trading days clean paper + Frank arms it |
| G | Meta-model + regime weights activation | §4.2 | 🧭 designed only | data thresholds (E5/E7 gates) + health_check READY verdicts |
| H | Factor redesign (contingency) | gate-a-decision-tree Branch 3 menu | dormant (Branch 1 hit) | only if edge decays |
| I | Mission Control (daily HTML report, notification tiers, slippage loop) | extensions-mission-control-factor-factory §I | 🧭 designed | build after E; I3 after F |
| J | Factor Factory (candidate protocol + seed backlog + negative control) | extensions-mission-control-factor-factory §J | 🧭 designed | build after E+I1; ≤2 candidates/month |

**Standing decisions log:** D1 insider→log-only ✅, D2 universe→~60 ✅, D3 num_paths→7 ✅ (Frank, 2026-07-11). **Branch 2 adopted 2026-07-19** (Frank delegated "do what's best"; post-C1 IC +0.061 falls squarely in the pre-agreed [0.03, 0.08) band): Kronos stays on but demoted from primary hope; live funding deferred ~60 run_dates; Robinhood funding waits for Frank's paycheck regardless. Open for Frank: vol_context kill (IC −0.176 FDR-significant), threshold re-tune question (OOS winner 0.65/0.30 vs production 0.70/0.45 — recommend: leave until 60 fresh run_dates, then walk-forward re-sweep).

---

## 3. Operating calendar (once B–E land)

| Cadence | What | Who/how |
|---|---|---|
| Daily 4:30pm ET | pipeline.py (~60 names, ~25 min) → proposals + brief | Task Scheduler (exists) |
| 5×/day Mon–Fri | intraday scan → paper ledger | Task Scheduler (exists) |
| Friday 4pm | health_check.py → kill/promote/ready verdicts | wire into weekly audit slot (E8) |
| Monthly | Frank + Claude: review health verdicts, act on KILL/PROMOTE candidates | manual session |
| ~+2wk from E | dispersion threshold tune (E2 Step 4); intraday kill verdict | health_check flags when ready |
| ~+60 run_dates | congress/insider promotion; regime fit unlock; Kelly activation review; threshold re-sweep | health_check flags |
| ~+1000 meta rows | meta-model training task (§4.2) | health_check flags TRAIN-READY |
| Yearly | macro_calendar.json refresh (BLS/Fed dates) | manual |

---

## 4. Design-level specs for future phases (build later, design frozen now)

### 4.1 Phase F — rh_executor.py (live execution bridge)

**Purpose:** translate intraday paper decisions into real Robinhood orders on the agentic account (~$2k) after paper validation. Fidelity is out of scope permanently.

**Promotion criteria (ALL required before first live order):**
- ≥10 trading days of post-reset paper signals with clean ledger reconciliation (no stale fills, no phantom rows)
- Intraday health verdict ≠ SHUT OFF (rank IC ≥ 0.02, net paper PnL ≥ 0 over ≥20 closed trades)
- Frank explicitly arms it (config flag `live_armed: true` + funded account) — no code path may self-arm

**Architecture:**
- Consumes ONLY the paper ledger's decision stream (positions.csv writes become order intents) — zero new decision logic; executor is dumb by design
- Order intent file handshake: `ledger/intraday/order_queue.csv` (intent → ack → fill/reject), so pipeline and executor stay decoupled processes
- Safety rails (constants, not config — changing them = code review):
  - MAX_ORDER_DOLLARS = 1000, MAX_OPEN_POSITIONS = 2, MAX_DAILY_ORDERS = 6
  - Kill switch: `ledger/intraday/KILL` file existence halts everything, checked before every order
  - Dry-run default: `--live` flag required per invocation; otherwise logs intents only
  - Reconciliation on every start: RH account positions vs positions.csv; mismatch → halt + notify, never "fix" silently
  - Market orders only during regular session; no orders within 10 min of close except force-close
- Auth: credentials via Windows Credential Manager only (per global secrets rule); never in config/env-file committed to repo
- Every order + response appended to `ledger/intraday/live_orders.csv` (append-only audit trail)

**Explicitly NOT in scope:** options, margin, shorting, Fidelity, any order type beyond market/limit-at-close.

**Amendment 2026-07-11 — broker abstraction + venue options (Frank raised direct-RH connection):**
- rh_executor is renamed in design to a thin `BrokerAdapter` interface: `get_positions() / get_buying_power() / get_quote(t) / place_order(intent) / order_status(id)`. Implementations: `PaperAdapter` (current homemade ledger — stays forever as the measurement baseline), `RobinhoodAdapter`, optionally `AlpacaAdapter`. All safety rails (caps, KILL file, dry-run, reconciliation-halt, Frank-armed flag) live ABOVE the adapter — venue-independent, no adapter can bypass them.
- A live broker connection, when confirmed, is used for three READ purposes immediately (zero risk): real-time quotes (upgrade over delayed yfinance for intraday scans), account-state reads for reconciliation, order-status polls. Order PLACEMENT remains gated behind the full Phase F promotion criteria — a connection existing is not a reason to trade sooner. Claude never places orders directly under any configuration; execution runs only through the armed executor process or Frank manually.
- Venue options ranked (engineering fit, not advice on where to hold money):
  1. **Alpaca paper API** — purpose-built algo interface, free realistic simulated fills + free IEX real-time data. Best PAPER venue: running it in parallel with the homemade ledger gives two independent fill models, and divergence between them measures our fill-assumption error before any real dollar trades.
  2. **Robinhood** (Frank's agentic account) — the designated LIVE venue per account plan; API surface is the least algo-friendly of the three, hence the dumb-adapter design.
  3. **IBKR** — most robust API/fills; complexity overkill at $2k, revisit only if capital scales 10×.
- Data redundancy (risk-register item, promoted): add a quote-source fallback interface mirroring the adapter pattern — yfinance primary, Alpaca/Polygon free tier as backup — so a yfinance breakage degrades to WARN instead of halting scans.
- Options/derivatives remain out of scope at this capital size — leverage amplifies an edge measured in hundredths of Sharpe into account-level risk; not a venue question.

### 4.2 Phase G — Meta-model + regime activation

**Meta-model (unlocks at TRAIN-READY, ~1000 meta rows):**
- Input: `output/meta_dataset.csv` (E7 extractor; features frozen there)
- Model: logistic regression FIRST (interpretable baseline); gradient boosting only if logistic shows lift and Frank wants the extra complexity
- Validation: purged walk-forward CV (reuse learn_weights.py A6 machinery), FDR-checked lift over base rate; holdout = most recent 20% of run_dates, never trained on
- Activation shape: multiplier on position size (0.5–1.0× by predicted P(win)), NEVER a new BUY generator — meta-model can only shrink/veto, not add trades
- Ships in log-mode like everything else: predicted P logged per row for ≥20 run_dates before sizing activation

**Regime weights (unlocks at FIT-READY, ≥60 run_dates per bucket):**
- Buckets frozen at VIX <20 / ≥20 (E5) — do not add buckets without doubling the data requirement
- Activation: `factor_weights: "regime"` switches compute_score to per-bucket learned weights, each bucket independently passing A6's holdout gate
- Fallback: any bucket failing holdout → that bucket uses equal weights (never global learned weights)

### 4.3 Decommission criteria (designed now so nobody argues later)

- Intraday system: SHUT OFF verdict twice consecutively → disable Task Scheduler entries, archive configs, keep code
- Kronos: decay alarm + walk-forward IC < 0.05 for 2 consecutive monthly checks → demote to mu-input only (decision-tree Branch 3 option 3)
- Any activated signal: loses FDR significance 2 consecutive monthly checks → back to log-mode (auto-recommend, Frank confirms)

---

## 5. Dispatch runbook (copy-paste briefs for future sessions)

**General form:** one Sonnet executor per phase, background Agent, general-purpose. Always include: plan path, interpreter path, scoped-pytest note, exact-commit-message rule, stop condition, report format. Never dispatch two executors into the repo simultaneously.

### 5.1 Phase B–D (ACTIVE — only if current executor dies unrecoverably)
> You are the sole executor of a locked plan. Project: C:\Projects\investment-pipeline. Read C:\Projects\investment-pipeline\docs\superpowers\plans\2026-07-11-audit-fix-plan.md. Check `git log --oneline` to see which B/C/D task commits already exist; resume at the first missing task, execute through D6 in plan order. Obey all Executor ground rules. Context: interpreter C:\Users\frank\AppData\Local\Programs\Python\Python314\python.exe; scoped pytest only (bare pytest hits vendored Kronos/); factor_log historical duplicates are protected; backtest kronos mode needs --yes; B6 archives, never deletes. Report per-task DONE+hash, scoped suite line, post-C1 Kronos IC vs +0.191, B6 archive paths, assumption mismatches.

### 5.2 Phase E (dispatch after B–D report reviewed)
> You are the sole executor of a locked plan. Project: C:\Projects\investment-pipeline. Read C:\Projects\investment-pipeline\docs\superpowers\plans\2026-07-11-phase-e-alpha-upgrades.md. Prerequisites are met (audit plan Phases A–D complete — verify via git log; D1/D2/D3 approvals recorded in the plan). Execute E1 → E4 → E3 → E2 → E7 → E5 → E6 → E8. Between tasks run one live `pipeline.py` run and confirm the new columns/behavior appear in the next factor_log rows before proceeding. Hard boundaries section is binding: everything ships OFF or log-mode except E3's top-quartile BUY filter. Same interpreter/pytest/commit rules as the audit plan. Report per-task DONE+hash, suite line, first-live-run confirmation per task, assumption mismatches.

### 5.3 Phase F (DO NOT dispatch until §4.1 promotion criteria confirmed + Frank arms)
> Write the detailed task plan first (architect session, not executor): expand §4.1 of the MASTER-BLUEPRINT into a TDD task plan (same format as audit plan), then dispatch a Sonnet executor on it. The safety rails in §4.1 are frozen design — an executor may not weaken them.

### 5.4 Phase G (dispatch on health_check TRAIN-READY / FIT-READY verdicts)
> Same two-step: architect session expands §4.2 into a task plan; executor builds it. Meta-model activation and regime-weight activation each require their own Frank sign-off after log-mode evidence.

### 5.5 Phase I — Mission Control (dispatch after Phase E lands)
> You are the sole executor. Project: C:\Projects\investment-pipeline. Read C:\Projects\investment-pipeline\docs\superpowers\plans\2026-07-11-extensions-mission-control-factor-factory.md, Phase I sections I1+I2 only (I3 waits for Phase F). Build report_daily.py and notify.py per spec and acceptance criteria; wire into run_scheduler.py so report failure never blocks the pipeline. Same interpreter/pytest/commit discipline as prior phases. Report: DONE+hashes, a rendered report_latest.html from real data, proof pipeline survives a forced report crash.

### 5.6 Phase J — Factor Factory (dispatch after E and I1)
> You are the sole executor. Project: C:\Projects\investment-pipeline. Read C:\Projects\investment-pipeline\docs\superpowers\plans\2026-07-11-extensions-mission-control-factor-factory.md, Phase J. Build the candidates.json loader with protocol rules enforced in code, wire high_52w_proximity as the reference candidate in log mode, integrate with health_check verdicts and the report signal board. Acceptance test is binding: raw_score identical with and without log-mode candidates. Report: DONE+hashes, first candidate visible in factor_log next run.

### 5.7 Weekly ops session (any Claude session, low effort)
> Read output/health/health_latest.txt (or run `$PY health_check.py`). Summarize verdicts for Frank. If KILL/PROMOTE/READY candidates exist, present evidence + recommendation. Take no action without Frank's reply. Update MASTER-BLUEPRINT §2 status table if a phase landed.

---

## 6. Risk register (engineering, not market)

| Risk | Mitigation |
|---|---|
| Executor API stalls mid-phase (happened 2×) | per-task commits = cheap resume; runbook 5.1 resume brief; never batch commits |
| yfinance schema/rate changes | staleness checks (B5/N7), bulk fetch reduces call count, watchers have stale-cache fallbacks |
| Silent scoring regression | test suite guards factor math; health_check catches IC drift; factor_log columns never silently dropped (A2 read-merge-write) |
| Data-starved ML (313 rows) | every ML layer hard-gated on row counts in CODE, not judgment |
| Threshold overfit recurrence | OOS-only reporting is now structural in backtest.py; Gate A doctrine in every plan |
| Live executor runaway (Phase F) | constants-not-config caps, KILL file, dry-run default, reconciliation-halt, Frank-armed flag |
| Context loss between sessions | this blueprint + per-phase plans + /wrapup to Brain; runbook briefs are self-contained |

---

## 7. What "perfect" means here (engineer's note)

Perfect prediction is not on any roadmap — it doesn't exist. What this system is engineered toward: **honest measurement, disciplined activation, capped downside, compounding of a small real edge.** Current honest baseline: Kronos IC +0.191, OOS net Sharpe 0.53, one FDR-significant anti-factor flagged for removal. The moat is the measurement loop: most retail systems never learn their edge is fake; this one now finds out in weeks, automatically, every Friday at 4pm.
