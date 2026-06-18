"""
Deterministic unit tests for the scoring + backtest core.

These cover the pure logic the strategy rests on — performance statistics,
the information-coefficient calculation, the BUY/HOLD/REDUCE classifier, the
forecast-horizon math, the Kronos annualization cap, and cross-sectional factor
standardization. No network, no data files, no model inference: every input is
synthetic and every expected value is computed by hand.

Run:  pytest test_pipeline.py -v
"""
import math
import numpy as np
import pandas as pd
import pytest

from backtest import perf_stats, TRADING_WEEK
from ic_report import spearman_ic
from learn_weights import standardize_cross_sectional, FACTORS
from pipeline import action_label, horizon_days, annualize_kronos_mu


# ── perf_stats (Sharpe / CAGR / max drawdown) ────────────────────────────────

class TestPerfStats:
    def test_empty_series_is_zeroed(self):
        s = perf_stats(pd.Series([], dtype=float))
        assert s == {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    def test_all_nan_is_zeroed(self):
        s = perf_stats(pd.Series([np.nan, np.nan]))
        assert s == {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    def test_monotonic_gains_have_no_drawdown(self):
        # every weekly return positive -> equity never dips below its peak.
        # alternate two positive values so volatility (and thus Sharpe) is > 0.
        r = pd.Series([0.01, 0.02] * (TRADING_WEEK // 2))
        s = perf_stats(r)
        assert s["max_dd"] == pytest.approx(0.0, abs=1e-9)
        assert s["cagr"] > 0
        assert s["sharpe"] > 0

    def test_constant_return_one_year_cagr(self):
        # exactly TRADING_WEEK weeks of r -> 1 year -> CAGR == compounded return
        r = pd.Series([0.01] * TRADING_WEEK)
        s = perf_stats(r)
        expected_cagr = (1.01 ** TRADING_WEEK) - 1
        assert s["cagr"] == pytest.approx(expected_cagr, rel=1e-6)

    def test_zero_volatility_gives_zero_sharpe(self):
        # std == 0 -> guarded branch returns 0.0 sharpe, not inf/NaN
        r = pd.Series([0.0] * TRADING_WEEK)
        assert perf_stats(r)["sharpe"] == 0.0

    def test_drawdown_is_negative_after_a_loss(self):
        r = pd.Series([0.10, -0.50, 0.05])
        assert perf_stats(r)["max_dd"] < 0


# ── spearman_ic (rank information coefficient) ───────────────────────────────

class TestSpearmanIC:
    def test_too_few_points_returns_nan(self):
        pred = pd.Series(range(9))
        realized = pd.Series(range(9))
        assert math.isnan(spearman_ic(pred, realized))

    def test_perfect_rank_agreement_is_one(self):
        pred = pd.Series(range(20))
        realized = pd.Series(range(20))
        assert spearman_ic(pred, realized) == pytest.approx(1.0)

    def test_perfect_rank_inversion_is_minus_one(self):
        pred = pd.Series(range(20))
        realized = pd.Series(range(20)[::-1])
        assert spearman_ic(pred, realized) == pytest.approx(-1.0)

    def test_monotonic_transform_preserves_rank_ic(self):
        # IC is rank-based: a monotonic (exp) transform must not change it
        pred = pd.Series(np.linspace(0, 1, 15))
        realized = pd.Series(np.exp(np.linspace(0, 1, 15)))
        assert spearman_ic(pred, realized) == pytest.approx(1.0)


# ── action_label (entry-gate classifier) ─────────────────────────────────────

class TestActionLabel:
    def test_below_reduce_cutoff(self):
        assert action_label(0.44, regime=True) == "REDUCE"

    def test_reduce_cutoff_is_inclusive_at_045(self):
        # 0.45 is NOT < 0.45 -> not a REDUCE
        assert action_label(0.45, regime=True) == "HOLD"

    def test_strong_score_in_good_regime_buys(self):
        assert action_label(0.75, regime=True) == "BUY"

    def test_buy_cutoff_boundary(self):
        assert action_label(0.70, regime=True) == "BUY"
        assert action_label(0.699, regime=True) == "HOLD"

    def test_good_score_bad_regime_is_gated_to_hold(self):
        # regime is an entry gate: a BUY-worthy score in a bad regime never buys
        assert action_label(0.95, regime=False) == "HOLD"

    def test_midrange_is_hold(self):
        assert action_label(0.55, regime=True) == "HOLD"


# ── horizon_days (forecast horizon in trading days) ──────────────────────────

class TestHorizonDays:
    def test_daily_bars_are_one_to_one(self):
        assert horizon_days("1d", 10) == 10.0

    def test_hourly_candles_to_days(self):
        # docstring example: 24 hourly candles ~ 3.4 trading days (7 bars/day)
        assert horizon_days("1h", 24) == pytest.approx(24 / 7)

    def test_unknown_interval_falls_back_to_seven(self):
        assert horizon_days("nonsense", 14) == pytest.approx(14 / 7)


# ── annualize_kronos_mu (capped annualization) ───────────────────────────────

class TestAnnualizeKronosMu:
    def test_nonpositive_horizon_is_zero(self):
        assert annualize_kronos_mu(0.05, 0.0) == 0.0
        assert annualize_kronos_mu(0.05, -3.0) == 0.0

    def test_short_horizon_gain_is_capped(self):
        # compounding a 2% 10-day return to a year explodes -> clipped to +cap
        assert annualize_kronos_mu(0.02, 10.0, cap=0.60) == pytest.approx(0.60)

    def test_loss_is_capped_at_negative(self):
        assert annualize_kronos_mu(-0.05, 10.0, cap=0.60) == pytest.approx(-0.60)

    def test_within_cap_is_passed_through(self):
        # a tiny return over a full year stays well inside the cap
        out = annualize_kronos_mu(0.01, 252.0, cap=0.60)
        assert out == pytest.approx(0.01, abs=1e-9)
        assert -0.60 < out < 0.60


# ── standardize_cross_sectional (per-date factor z-scores) ───────────────────

class TestStandardize:
    def _frame(self, n_per_date):
        rows = []
        for d in ("2026-06-01", "2026-06-02"):
            for i in range(n_per_date):
                row = {"run_date": d}
                for f in FACTORS:
                    row[f] = float(i + 1)  # 1,2,3,... spread across tickers
                rows.append(row)
        return pd.DataFrame(rows)

    def test_zscores_are_mean_zero_per_date(self):
        out = standardize_cross_sectional(self._frame(3))
        for d, g in out.groupby("run_date"):
            assert g[FACTORS].mean().abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_fewer_than_three_names_dropped(self):
        # z-score needs 3+ names/date; 2 -> NaN -> dropped entirely
        out = standardize_cross_sectional(self._frame(2))
        assert out.empty

    def test_output_columns_preserved(self):
        out = standardize_cross_sectional(self._frame(4))
        assert set(FACTORS).issubset(out.columns)
        assert "run_date" in out.columns
