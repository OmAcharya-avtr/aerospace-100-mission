"""Known-answer, property and input-validation tests for the published baselines.

Every expected value in a known-answer test is hand-computed in the comment
above the assertion, from the published closed form, not copied from this
implementation's output.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cncast.baselines import (
    HV57_A0,
    HV57_RMS_WIND,
    bufton_wind,
    hufnagel_valley,
    hv57,
    rms_high_altitude_wind,
    slc_day,
    slc_night,
)

# ---------------------------------------------------------------- known answers


def test_hv57_at_ground_is_sum_of_two_terms() -> None:
    """Hand check, HV 5/7 at h = 0.

    high  = 0.00594 (21/27)^2 (1e-5 * 0)^10 e^0            = 0
    tropo = 2.7e-16 * e^0                                  = 2.7e-16
    ground= 1.7e-14 * e^0                                  = 1.7e-14
    total                                                  = 1.727e-14 m^-2/3
    """
    assert float(hv57(np.array([0.0]))[0]) == pytest.approx(1.727e-14, rel=1e-12)


def test_hv57_at_100_m() -> None:
    """Hand check, HV 5/7 at h = 100 m.

    ground = 1.7e-14 * e^-1        = 1.7e-14 * 0.36787944 = 6.2539505e-15
    tropo  = 2.7e-16 * e^-0.066667 = 2.7e-16 * 0.93551          = 2.5258866e-16
    high   = 0.00594*(21/27)^2*(1e-3)^10*e^-0.1 = 3.6e-3*1e-30*0.905 ~ 3.3e-33
    total                                                  = 6.5065392e-15
    """
    assert float(hv57(np.array([100.0]))[0]) == pytest.approx(6.5065392e-15, rel=1e-6)


def test_hv57_high_altitude_term_at_10km() -> None:
    """Hand check of the H-V high-altitude term alone at h = 10 km, v = 21 m/s.

    0.00594 * (21/27)^2 * (1e-5 * 1e4)^10 * e^-10
      = 0.00594 * 0.6049383 * (0.1)^10 * 4.5399930e-5
      = 0.00594 * 0.6049383 * 1e-10 * 4.5399930e-5
      = 1.6314e-17 m^-2/3
    The tropopause term there is 2.7e-16 e^(-20/3) = 3.436e-19, the ground term
    is ~1e-57, so the total is 1.6657e-17.
    """
    total = float(hv57(np.array([10_000.0]))[0])
    assert total == pytest.approx(1.6657e-17, rel=2e-4)


def test_hv_parameters_are_hv57_defaults() -> None:
    """hv57 is exactly hufnagel_valley(v = 21 m/s, A = 1.7e-14)."""
    h = np.geomspace(1.0, 20_000.0, 50)
    assert np.allclose(hv57(h), hufnagel_valley(h, HV57_RMS_WIND, HV57_A0), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("h", "expected"),
    [
        # SLC-Day branch 2: 3.13e-13 / 100^1.05 = 3.13e-13 / 125.89254 = 2.4862468e-15
        (100.0, 2.4862468e-15),
        # SLC-Day branch 3 is the constant 1.3e-15 on 240-880 m
        (500.0, 1.3e-15),
        # SLC-Day branch 4: 8.87e-7 / 2000^3 = 8.87e-7 / 8e9 = 1.108750e-16
        (2000.0, 1.10875e-16),
        # SLC-Day branch 5: 2.0e-16 / sqrt(1e4) = 2.0e-16 / 100 = 2.0e-18
        (10000.0, 2.0e-18),
    ],
)
def test_slc_day_known_values(h: float, expected: float) -> None:
    """SLC-Day piecewise branches against the published closed forms."""
    assert float(slc_day(np.array([h]))[0]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("h", "expected"),
    [
        # constant surface branch
        (10.0, 8.4e-15),
        # 2.87e-12 / 50^2 = 2.87e-12 / 2500 = 1.148e-15
        (50.0, 1.148e-15),
        # constant 2.5e-16 on 110-1500 m
        (500.0, 2.5e-16),
        # 8.87e-7 / 3000^3 = 8.87e-7 / 2.7e10 = 3.2851852e-17
        (3000.0, 3.2851852e-17),
        # 2.0e-16 / sqrt(1e4) = 2.0e-18
        (10000.0, 2.0e-18),
    ],
)
def test_slc_night_known_values(h: float, expected: float) -> None:
    """SLC-Night piecewise branches against the published closed forms."""
    assert float(slc_night(np.array([h]))[0]) == pytest.approx(expected, rel=1e-6)


def test_slc_profiles_are_zero_above_their_ceilings() -> None:
    """The published SLC fits are defined as identically zero above their tops."""
    assert float(slc_day(np.array([20_501.0]))[0]) == 0.0
    assert float(slc_day(np.array([20_499.0]))[0]) > 0.0
    assert float(slc_night(np.array([20_001.0]))[0]) == 0.0
    assert float(slc_night(np.array([19_999.0]))[0]) > 0.0


def test_slc_day_exceeds_slc_night_in_the_boundary_layer() -> None:
    """Daytime surface heating: SLC-Day > SLC-Night below ~1 km, by design.

    At 10 m the published constants give 1.7e-14 / 8.4e-15 = 2.02.
    """
    h = np.array([1.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
    assert np.all(slc_day(h) > slc_night(h))
    assert float(slc_day(np.array([10.0]))[0] / slc_night(np.array([10.0]))[0]) == pytest.approx(
        2.0238, rel=1e-3
    )


def test_bufton_wind_peaks_at_jet_stream_altitude() -> None:
    """V(h) = w_g + 30 exp(-((h-9400)/4800)^2): peak w_g + 30 at h = 9400 m."""
    assert float(bufton_wind(np.array([9400.0]), 5.0)[0]) == pytest.approx(35.0, rel=1e-12)
    # at h = 9400 + 4800 the exponent is -1: 5 + 30 * e^-1 = 5 + 11.036 = 16.036
    assert float(bufton_wind(np.array([14200.0]), 5.0)[0]) == pytest.approx(16.0360, rel=1e-4)


def test_rms_wind_increases_with_ground_wind() -> None:
    """The 5-20 km rms of the Bufton profile is monotone in the ground wind."""
    values = [rms_high_altitude_wind(w) for w in (0.0, 5.0, 10.0, 20.0)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    # w_g = 0 leaves only the jet-stream bump; the rms over 5-20 km is ~18.7 m/s
    assert values[0] == pytest.approx(18.68, rel=1e-3)


# ---------------------------------------------------------------- physics properties


def test_cn2_decreases_with_altitude_through_the_boundary_layer() -> None:
    """Physical plausibility: Cn^2 falls steeply from the surface upward.

    Checked on 5 m - 5 km, i.e. above the surface layer and below the H-V
    jet-stream bump.  See ``validation/VALIDATION.md`` §3: HV 5/7 is NOT monotone
    over the whole column - the high-altitude term deliberately produces a bump
    peaking near 9.8 km - so a global monotonicity assertion would be wrong.
    """
    h = np.geomspace(5.0, 5000.0, 500)
    assert np.all(np.diff(hv57(h)) < 0.0)
    assert np.all(np.diff(hufnagel_valley(h, 30.0, 5e-15)) < 0.0)


def test_hv_has_a_jet_stream_bump_near_10_km() -> None:
    """The high-altitude term makes Cn^2 rise again between ~6 and ~10 km."""
    h = np.geomspace(6000.0, 20000.0, 4000)
    c = hv57(h)
    peak = float(h[int(np.argmax(c))])
    assert 8000.0 < peak < 11000.0
    assert float(hv57(np.array([10_000.0]))[0]) > float(hv57(np.array([6000.0]))[0])


def test_hv_column_drops_by_orders_of_magnitude() -> None:
    """Cn^2(20 km) is more than 1e3 times smaller than Cn^2(300 m) for HV 5/7."""
    low = float(hv57(np.array([300.0]))[0])
    high = float(hv57(np.array([19_000.0]))[0])
    assert low / high > 100.0


def test_hv_ground_term_scales_linearly_with_a0() -> None:
    """At h = 0 the ground term is A exactly, so doubling A shifts the total by A."""
    a = 1.0e-14
    base = float(hufnagel_valley(np.array([0.0]), 21.0, a)[0])
    doubled = float(hufnagel_valley(np.array([0.0]), 21.0, 2 * a)[0])
    assert doubled - base == pytest.approx(a, rel=1e-9)


@given(
    v=st.floats(min_value=1.0, max_value=60.0),
    h=st.floats(min_value=7000.0, max_value=12000.0),
)
@settings(max_examples=40, deadline=None)
def test_high_altitude_term_scales_as_v_squared(v: float, h: float) -> None:
    """Algebraic identity: the H-V high-altitude term is proportional to v^2.

    Isolated by setting A = 0 and subtracting the (v-independent) tropopause term.
    """
    ha = np.array([h])
    tropo = 2.7e-16 * np.exp(-h / 1500.0)
    one = float(hufnagel_valley(ha, 1.0, 0.0)[0]) - tropo
    scaled = float(hufnagel_valley(ha, v, 0.0)[0]) - tropo
    assert scaled == pytest.approx(one * v**2, rel=1e-9)


@given(h=st.floats(min_value=0.0, max_value=25_000.0))
@settings(max_examples=50, deadline=None)
def test_all_baselines_are_non_negative(h: float) -> None:
    """Cn^2 is a variance-like quantity: never negative, always finite."""
    for fn in (hv57, slc_day, slc_night):
        value = float(fn(np.array([h]))[0])
        assert value >= 0.0
        assert np.isfinite(value)


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize("fn", [hv57, slc_day, slc_night])
def test_negative_altitude_rejected(fn) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        fn(np.array([-1.0]))


@pytest.mark.parametrize("fn", [hv57, slc_day, slc_night])
def test_non_finite_altitude_rejected(fn) -> None:
    with pytest.raises(ValueError, match="finite"):
        fn(np.array([np.nan]))


@pytest.mark.parametrize("fn", [hv57, slc_day, slc_night])
def test_empty_altitude_rejected(fn) -> None:
    with pytest.raises(ValueError, match="at least one"):
        fn(np.array([]))


def test_negative_wind_rejected() -> None:
    with pytest.raises(ValueError, match="rms_wind_m_s"):
        hufnagel_valley(np.array([100.0]), -1.0, 1.7e-14)
    with pytest.raises(ValueError, match="ground_wind_m_s"):
        bufton_wind(np.array([100.0]), -0.1)


def test_negative_a0_rejected() -> None:
    with pytest.raises(ValueError, match="a0"):
        hufnagel_valley(np.array([100.0]), 21.0, -1e-15)


def test_rms_wind_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="n_points"):
        rms_high_altitude_wind(5.0, n_points=2)


def test_shape_is_preserved() -> None:
    h = np.linspace(0.0, 1000.0, 7).reshape(7, 1)
    assert hv57(h).shape == (7, 1)
    assert slc_day(h).shape == (7, 1)
