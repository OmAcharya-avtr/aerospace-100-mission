"""Power-law capacity-fade baseline."""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.baselines import DegradationFitError
from degradepredict.limits import LimitSpec
from degradepredict.physics import (
    DEFAULT_FADE_EXPONENT,
    FADE_EXPONENT_GRID,
    powerlaw_fade_baseline,
)

LIMIT = LimitSpec(value=80.0, direction="below", unit="percent of BOL", atol=1e-9)


def test_default_exponent_is_the_cited_value():
    # Wang et al. (2011), Journal of Power Sources 196(8):3942-3948, z = 0.55.
    assert DEFAULT_FADE_EXPONENT == 0.55
    assert 0.5 in set(np.round(FADE_EXPONENT_GRID, 2))
    assert 0.55 in set(np.round(FADE_EXPONENT_GRID, 2))


def test_recovers_a_hand_calculated_crossing():
    # C(t) = 100 (1 - B t**0.55) reaches 80 when B t**0.55 = 0.2.
    # Choose B so that the crossing is at exactly 1000 d:
    #   B = 0.2 / 1000**0.55 = 0.2 / 44.6684 = 0.0044774...
    z = 0.55
    b = 0.2 / 1000.0**z
    t = np.arange(60, dtype=float) * 7.0
    y = 100.0 * (1.0 - b * t**z)
    y = y + np.random.default_rng(1).normal(0.0, 1e-6, size=t.size)
    fit = powerlaw_fade_baseline(t, y, LIMIT, z=z)
    assert fit.crossing_days == pytest.approx(1000.0, rel=1e-4)
    assert fit.z == z
    assert fit.model == "powerlaw-fade"


def test_fit_is_linear_in_its_parameters_so_the_slope_is_minus_c0_times_b():
    z = 0.55
    b = 0.2 / 1000.0**z
    t = np.arange(40, dtype=float) * 7.0
    y = 100.0 * (1.0 - b * t**z) + np.random.default_rng(2).normal(0.0, 1e-6, size=t.size)
    fit = powerlaw_fade_baseline(t, y, LIMIT, z=z)
    assert fit.intercept == pytest.approx(100.0, rel=1e-5)
    assert fit.slope == pytest.approx(-100.0 * b, rel=1e-4)


def test_profiled_exponent_picks_a_grid_value_and_fits_better_in_sample():
    z_true = 0.70
    b = 0.2 / 900.0**z_true
    t = np.arange(50, dtype=float) * 7.0
    y = 100.0 * (1.0 - b * t**z_true) + np.random.default_rng(3).normal(0.0, 0.1, size=t.size)
    profiled = powerlaw_fade_baseline(t, y, LIMIT, z=None)
    fixed = powerlaw_fade_baseline(t, y, LIMIT, z=DEFAULT_FADE_EXPONENT)
    assert profiled.z in set(np.round(FADE_EXPONENT_GRID, 2))
    assert profiled.sigma <= fixed.sigma
    assert abs(profiled.crossing_days - 900.0) < abs(fixed.crossing_days - 900.0)


def test_profiled_exponent_reports_when_no_grid_value_works():
    t = np.arange(30, dtype=float) * 7.0
    y = 95.0 + 0.01 * t + np.random.default_rng(4).normal(0.0, 0.05, size=t.size)
    with pytest.raises(DegradationFitError, match="no fade exponent"):
        powerlaw_fade_baseline(t, y, LIMIT, z=None)


def test_rejects_a_non_positive_exponent():
    t = np.arange(20, dtype=float) * 7.0
    y = 100.0 - 0.01 * t + np.random.default_rng(5).normal(0.0, 0.1, size=t.size)
    with pytest.raises(ValueError, match="z must be finite and positive"):
        powerlaw_fade_baseline(t, y, LIMIT, z=-0.5)


def test_fade_model_with_z_one_matches_the_linear_baseline():
    """z = 1 degenerates to the straight-line fit; the two must agree."""
    from degradepredict.baselines import linear_baseline

    t = np.arange(40, dtype=float) * 7.0
    y = 100.0 - 0.02 * t + np.random.default_rng(6).normal(0.0, 0.3, size=t.size)
    a = powerlaw_fade_baseline(t, y, LIMIT, z=1.0)
    b = linear_baseline(t, y, LIMIT)
    assert a.crossing_days == pytest.approx(b.crossing_days, rel=1e-12)
    assert a.fieller_g == pytest.approx(b.fieller_g, rel=1e-12)
