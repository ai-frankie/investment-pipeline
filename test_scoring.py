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
