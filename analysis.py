# Analysis script to check for bugs and data leakage

import sys
import re
from pathlib import Path

files_to_check = {
    "pipeline.py": [
        (405, 414, "Kronos annualization with cap masks outliers"),
        (244, 247, "Forecast edge clipping silently maxes at 1.0"),
        (704, 726, "Factor log append without date offset / lookahead bias"),
        (101, 119, "Vol scalar initialization miss when targets empty"),
        (122, 133, "Earnings date timezone mismatch silent fail"),
        (383, 398, "Action label regime gate inconsistent (new vs hold)"),
    ],
    "kronos_forecast.py": [
        (156, 164, "Path aggregation element-wise mean creates synthetic path"),
        (129, 130, "Forecast reuse check never validates asof parameter"),
    ],
    "ledger.py": [
        (125, 125, "Share sizing: int() truncation loses fractional shares"),
        (113, 113, "Pending fill date comparison: string vs datetime, weekends"),
        (96, 104, "Starting cash parameter shadowed by state.json load"),
    ],
    "run_scheduler.py": [
        (50, 81, "Ollama start timeout 3s too short for service startup"),
    ],
    "daily_brief.py": [
        (31, 32, "latest_proposals silently returns None if output/ missing"),
        (60, 63, "Ollama timeout 300s silently fails without retry"),
    ],
}

print("=== STATIC ANALYSIS: BUGS + DATA LEAKAGE ===\n")

for filename, issues in files_to_check.items():
    print(f"\n{filename}")
    print("-" * 60)
    for start, end, desc in issues:
        print(f"  Line {start}-{end}: {desc}")

print("\n\n=== CRITICAL FINDINGS ===\n")

findings = [
    ("pipeline.py", 405, "bug", "Kronos annualization caps at 0.60 (line 414), hiding 200x+ returns", 
     "Use log-scale scaling or percentile clipping instead of hard cap"),
    
    ("pipeline.py", 244, "bug", "Forecast edge clipped to 1.0 (line 247); true edge quality masked",
     "Remove min(1.0, ...) or use softmax scoring"),
    
    ("pipeline.py", 704, "bug", "factor_log.csv records decisions WITH next-day fills; creates 1-day lookahead bias in training",
     "Separate decision log (score time) from fill log (fill time); align IC backtest to score date"),
    
    ("kronos_forecast.py", 156, "bug", "Path aggregation (line 164) creates synthetic mid-path never in sampled distribution",
     "Keep sampled paths separate; do not average; use quantiles or full distribution for scoring"),
    
    ("ledger.py", 125, "bug", "Share sizing uses int() truncation (line 125), silently loses capital",
     "Use round(shares, 0) or track fractional shares separately"),
    
    ("ledger.py", 113, "bug", "Pending fill date comparison is string-based (line 113), fills across weekends",
     "Convert to datetime, use trading_day_offset or calendar checks"),
    
    ("pipeline.py", 383, "risk", "Action label regime gate only blocks NEW BUY, not HOLD->BUY (inconsistent)",
     "Add regime check to HOLD action or remove for consistency"),
    
    ("pipeline.py", 122, "risk", "Earnings blackout fails silently if timezone mismatch (line 128-133)",
     "Add explicit timezone normalization before comparison"),
    
    ("ledger.py", 96, "risk", "Starting cash parameter ignored if state.json exists (line 104)",
     "Reset state.json explicitly or pass override flag"),
    
    ("pipeline.py", 101, "risk", "Vol scalar silently skipped if optimizer returns empty dict",
     "Add log warning; raise if all tickers filtered out"),
]

for fname, line, sev, problem, fix in findings:
    print(f"{fname}:{line}: {sev}: {problem}")
    print(f"   -> {fix}\n")

