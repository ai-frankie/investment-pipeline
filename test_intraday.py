"""
Unit tests for intraday_pipeline.py ledger plumbing and bar handling.

Covers PDT-tracker persistence, corrupt-state recovery, position sizing,
and partial-bar volume scoring. No network.

Run:  pytest test_intraday.py -v
"""
import json
from datetime import timedelta

import numpy as np
import pandas as pd

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


def test_size_position_basic():
    assert ip._size_position(100.0, 0.40, 2000.0) == 8   # 2000*0.40/100


def test_size_position_price_exceeds_alloc():
    assert ip._size_position(900.0, 0.40, 2000.0) == 0   # caller must skip


def test_size_position_bad_price():
    assert ip._size_position(None, 0.40, 2000.0) is None
    assert ip._size_position(0.0, 0.40, 2000.0) is None


def test_volume_surge_ignores_partial_bar(monkeypatch):
    # last bar is still forming (tiny volume); second-to-last is 2x average.
    # Score must reflect the COMPLETED bar (>0), not the partial one (0).
    n = 30
    ts = pd.date_range("2026-07-08 09:30", periods=n, freq="h")
    vol = np.full(n, 1000.0)
    vol[-2] = 2000.0   # completed bar: 2x baseline
    vol[-1] = 10.0     # partial bar: near-zero so far
    df = pd.DataFrame({"timestamps": ts,
                       "close": np.linspace(100.0, 101.0, n),
                       "volume": vol})
    fake_now = ts[-1].to_pydatetime() + timedelta(minutes=30)  # bar 50% elapsed
    monkeypatch.setattr(ip, "now_et", lambda: fake_now)
    assert ip._score_volume_surge(df) > 0.0
