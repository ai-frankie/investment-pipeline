"""
Unit tests for intraday_pipeline.py ledger plumbing.

Covers PDT-tracker persistence and corrupt-state recovery. No network.

Run:  pytest test_intraday.py -v
"""
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


def test_size_position_basic():
    assert ip._size_position(100.0, 0.40, 2000.0) == 8   # 2000*0.40/100


def test_size_position_price_exceeds_alloc():
    assert ip._size_position(900.0, 0.40, 2000.0) == 0   # caller must skip


def test_size_position_bad_price():
    assert ip._size_position(None, 0.40, 2000.0) is None
    assert ip._size_position(0.0, 0.40, 2000.0) is None
