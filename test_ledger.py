"""
Unit tests for ledger.py session-aware fill logic.

Covers the weekend no-op guard and the staleness bound that prevents a
missed scheduler run from filling days-old proposals at today's open.
No network: mark() must bail out before any yfinance call on a weekend.

Run:  pytest test_ledger.py -v
"""
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
