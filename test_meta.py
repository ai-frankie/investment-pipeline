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
