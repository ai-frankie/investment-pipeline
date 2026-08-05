# Gate A Decision Tree — what to do with the honest numbers

Written before Gate A results exist, so the decision framework can't be bent to fit the numbers after the fact.

## Inputs (from notes/2026-07-11-revalidation-results.md when executor finishes)

- `IC_new`: Kronos daily IC, next-open anchored, non-overlapping windows (old claim: +0.145)
- `Sharpe_oos_net`: sweep OOS Sharpe net of 10 bps (old claim: 0.71, in-sample gross)
- Per-factor FDR significance flags

## Branches

### Branch 1 — Edge holds: IC_new ≥ 0.08 AND Sharpe_oos_net ≥ 0.50
Old numbers were inflated but the signal is real.
- Proceed: Phase B → C → D → E exactly as planned, no changes.
- Robinhood funding timeline unchanged (paper clock restarts at Phase B6 anyway).

### Branch 2 — Edge weakened but alive: IC_new in [0.03, 0.08) OR Sharpe_oos_net in [0.25, 0.50)
Signal exists but thin — inflation was mostly measurement artifacts.
- Proceed Phase B → C → D (correctness is unconditional).
- Phase E priority reshuffle: E3 (rank scoring) and E4 (alt-data measurement) become the main alpha hopes, not Kronos. E2 dispersion gate matters more (filter thin signal harder).
- Sizing stance: keep kelly_fraction at 0 until rank/alt-data ICs are measured. Live funding decision deferred a full validation cycle (~60 run_dates).

### Branch 3 — Edge gone: IC_new < 0.03 OR not FDR-significant OR Sharpe_oos_net < 0.25
The validated edge was measurement error. Honest and useful to know now, not after funding.
- STILL run Phase B → D: correct plumbing is required to measure anything, including replacements.
- Phase E runs in measurement-only posture: E3/E4/E1 ship (they gather evidence), E2 activation moot, E6 stays dormant indefinitely.
- Live trading: no-go until a replacement signal validates (Frank's call, but the data won't support funding).
- Factor redesign menu (separate future plan, judged then — pick by what the per-factor FDR flags show):
  1. **Horizon shift**: re-test factors at 21d / weekly-rotation horizon — momentum/trend factors often work at horizons the 10d test misses. Cheap: same factor_log, different label horizon in the fixed IC harness.
  2. **Cross-sectional only**: drop absolute scoring; long top-decile rank vs universe (E3 already builds the machinery). Rank IC is the deliverable to watch.
  3. **Kronos demotion**: remove forecast_edge/path_consistency from the score, keep Kronos only as optimizer mu input (its weakest claim). Score = price/volume/alt-data factors only.
  4. **Alt-data first**: if insider/congress/contract ICs show FDR-significant signal while price factors don't, invert the architecture — alt-data events as primary trigger, price factors as filters.
  5. **Passive fallback**: if nothing validates after redesign cycle, the honest answer is the boring one — the system becomes a risk-managed monitoring tool, not an alpha engine. Stated so it's on the table, not implied away.

### Branch 4 — Executor blocked/failed
- I get the completion/failure notification, read the report, fix the blocking assumption myself or re-dispatch with corrected instructions. Plan-assumption mismatches listed in the executor's report get patched into the plan doc first (plan stays the single source of truth).

## Standing rule

Whatever branch: no threshold, no factor, no sizing lever gets tuned on the same data window that reports its performance. Gate A doctrine is permanent.
