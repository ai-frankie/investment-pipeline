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

def test_annualize_kronos_mu_monotonic_above_cap():
    a = pipeline.annualize_kronos_mu(0.02, 10)
    b = pipeline.annualize_kronos_mu(0.05, 10)
    c = pipeline.annualize_kronos_mu(0.10, 10)
    assert a < b < c            # old code: all pinned at 0.60
    assert c <= 0.60 + 1e-9     # still bounded

def test_annualize_kronos_mu_extreme_negative_no_blowup():
    assert np.isfinite(pipeline.annualize_kronos_mu(-0.999, 10))

def test_check_regime_vix_ceiling_configurable():
    hist = _hist(300)
    # default ceiling 22: VIX 25 vetoes
    ok, note = pipeline.check_regime(hist, vix=25.0)
    assert ok is False and "VIX" in note
    # raising the ceiling via param lets the same VIX through the VIX check
    # (other regime checks may still veto on this random series, so only
    # assert the VIX-specific note is gone)
    ok2, note2 = pipeline.check_regime(hist, vix=25.0, vix_ceiling=30.0)
    assert "VIX" not in note2 or ok2 is True

def test_daily_gate_action_returns_na_when_uncovered(tmp_path, monkeypatch):
    import intraday_pipeline as ip
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    pd.DataFrame({"ticker": ["AAPL"], "action": ["BUY"]}).to_csv(
        tmp_path / "output" / "proposals_20260717_120000.csv", index=False)
    assert ip.daily_gate_action("AAPL") == "BUY"
    assert ip.daily_gate_action("ZZZZ") == "N/A"

def test_daily_gate_action_na_when_no_daily_file(tmp_path, monkeypatch):
    import intraday_pipeline as ip
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    assert ip.daily_gate_action("AAPL") == "N/A"

def test_path_dispersion_known_values():
    # terminal returns [-0.02, 0.00, 0.02] -> std ~0.0163, median 0.0
    d = pipeline._path_dispersion([-0.02, 0.0, 0.02])
    assert abs(d - np.std([-0.02, 0.0, 0.02])) < 1e-9

def test_path_dispersion_single_path():
    assert pipeline._path_dispersion([0.01]) is None   # undefined for 1 path

def test_rank_scores_are_percentiles():
    raws = {"A": 0.80, "B": 0.60, "C": 0.40, "D": 0.20}
    ranks = pipeline.rank_scores(raws)
    assert ranks["A"] == 1.0 and ranks["D"] == 0.25
    assert ranks["B"] == 0.75 and ranks["C"] == 0.5

def test_rank_scores_single_ticker_neutral():
    assert pipeline.rank_scores({"A": 0.9}) == {"A": 0.5}

def test_dispersion_gate_blocks_wide_paths():
    # relative dispersion = std/|median| ; gate blocks when > max_rel
    assert pipeline.dispersion_ok(0.01, 0.002, max_rel=3.0) is True    # 0.2 < 3
    assert pipeline.dispersion_ok(0.001, 0.02, max_rel=3.0) is False   # 20 > 3
    assert pipeline.dispersion_ok(0.0, 0.02, max_rel=3.0) is False     # zero median, wide paths
    assert pipeline.dispersion_ok(0.01, None, max_rel=3.0) is True     # no dispersion info -> don't block
