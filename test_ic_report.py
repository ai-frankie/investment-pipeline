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
