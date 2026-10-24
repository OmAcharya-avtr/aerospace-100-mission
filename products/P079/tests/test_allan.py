"""The overlapping Allan deviation, its intervals, and the noise identifier."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.allan import (
    allan_table,
    equivalent_degrees_of_freedom,
    fit_log_slope,
    identify_noise,
    octave_taus,
    overlapping_allan_variance,
)
from timetransfer.errors import ConfigurationError
from timetransfer.noise import NOISE_KINDS, synthetic_phase, theoretical_adev, theory_slope

LEVELS = {"white_pm": 1.0e-9, "white_fm": 1.0e-20, "random_walk_fm": 1.0e-24}


def test_allan_variance_of_a_linear_phase_ramp_is_zero_to_rounding() -> None:
    # A pure frequency offset is a straight line in phase, whose second
    # differences vanish identically in exact arithmetic. In floating point the
    # claim is "zero to within rounding", which is what is asserted.
    x = 1.0e-6 * np.arange(500, dtype=np.float64)
    avar, n_terms = overlapping_allan_variance(x, 1.0, 4)
    assert n_terms == 500 - 8
    assert avar == pytest.approx(0.0, abs=1e-30)


def test_allan_variance_known_answer_by_hand() -> None:
    # x = [0, 0, 3, 0, 0], tau0 = 1, m = 1. Second differences at i=0,1,2:
    #   x2 - 2 x1 + x0 = 3 - 0 + 0 =  3
    #   x3 - 2 x2 + x1 = 0 - 6 + 0 = -6
    #   x4 - 2 x3 + x2 = 0 - 0 + 3 =  3
    # sum of squares = 9 + 36 + 9 = 54; n_terms = 3; tau = 1
    # avar = 54 / (2 * 3 * 1) = 9.0
    x = np.asarray([0.0, 0.0, 3.0, 0.0, 0.0])
    avar, n_terms = overlapping_allan_variance(x, 1.0, 1)
    assert n_terms == 3
    assert avar == pytest.approx(9.0, rel=1e-14)


def test_allan_variance_is_invariant_under_a_phase_offset() -> None:
    x = synthetic_phase("white_fm", 400, tau0_s=1.0, level=1e-20, seed=1)
    a, _ = overlapping_allan_variance(x, 1.0, 2)
    b, _ = overlapping_allan_variance(x + 1.0e-3, 1.0, 2)
    assert b == pytest.approx(a, rel=1e-9)


def test_allan_variance_scales_quadratically_with_amplitude() -> None:
    x = synthetic_phase("white_pm", 400, tau0_s=1.0, level=1e-9, seed=2)
    a, _ = overlapping_allan_variance(x, 1.0, 3)
    b, _ = overlapping_allan_variance(5.0 * x, 1.0, 3)
    assert b == pytest.approx(25.0 * a, rel=1e-12)


@pytest.mark.parametrize("m", [0, -1, 250])
def test_allan_variance_rejects_an_invalid_averaging_factor(m: int) -> None:
    x = np.zeros(500)
    with pytest.raises(ConfigurationError, match="averaging factor"):
        overlapping_allan_variance(x, 1.0, m)


def test_allan_variance_rejects_short_and_non_finite_input() -> None:
    with pytest.raises(ConfigurationError, match="at least 3 samples"):
        overlapping_allan_variance(np.zeros(2), 1.0, 1)
    with pytest.raises(ConfigurationError, match="non-finite"):
        overlapping_allan_variance(np.asarray([0.0, np.nan, 1.0, 2.0]), 1.0, 1)
    with pytest.raises(ConfigurationError, match="tau0_s"):
        overlapping_allan_variance(np.zeros(10), 0.0, 1)


def test_edf_white_pm_known_answer() -> None:
    # (N+1)(N-2m) / (2(N-m)) at N=100, m=1: 101*98 / (2*99) = 9898/198
    assert equivalent_degrees_of_freedom("white_pm", 100, 1) == pytest.approx(
        9898.0 / 198.0, rel=1e-12
    )


def test_edf_white_fm_known_answer() -> None:
    # (3(N-1)/(2m) - 2(N-2)/N) * 4m^2/(4m^2+5) at N=100, m=1:
    #   3*99/2 = 148.5 ; 2*98/100 = 1.96 ; 148.5 - 1.96 = 146.54
    #   4/(4+5) = 0.444444... ; 146.54 * 4/9 = 65.1288888...
    assert equivalent_degrees_of_freedom("white_fm", 100, 1) == pytest.approx(
        146.54 * 4.0 / 9.0, rel=1e-12
    )


def test_edf_random_walk_fm_known_answer() -> None:
    # ((N-2)/m) * ((N-1)^2 - 3m(N-1) + 4m^2) / (N-3)^2 at N=100, m=1:
    #   98 * (9801 - 297 + 4) / 9409 = 98 * 9508 / 9409
    assert equivalent_degrees_of_freedom("random_walk_fm", 100, 1) == pytest.approx(
        98.0 * 9508.0 / 9409.0, rel=1e-12
    )


def test_edf_flicker_fm_has_two_branches_and_the_first_is_always_clipped() -> None:
    # m = 1 : 2(N-2)/(2.3N - 4.9) = 196/225.1 = 0.87072..., which is below one
    # degree of freedom. The formula tends to 2/2.3 = 0.8696 as N grows, so the
    # flicker-FM branch at m = 1 is *always* below one and is always clipped to
    # 1.0 by equivalent_degrees_of_freedom. That is a property of the
    # simplified formula, it is reported here rather than worked around, and it
    # makes the m = 1 interval for flicker FM the widest this package can
    # produce.
    assert 196.0 / 225.1 < 1.0
    assert equivalent_degrees_of_freedom("flicker_fm", 100, 1) == 1.0
    assert equivalent_degrees_of_freedom("flicker_fm", 10_000, 1) == 1.0
    # m = 2 : 5N^2/(4m(N+3m)) = 50000/(8*106)
    assert equivalent_degrees_of_freedom("flicker_fm", 100, 2) == pytest.approx(
        50000.0 / (8.0 * 106.0), rel=1e-12
    )


def test_edf_is_floored_at_one() -> None:
    assert equivalent_degrees_of_freedom("white_pm", 11, 5) >= 1.0


@pytest.mark.parametrize(
    ("kind", "n", "m", "match"),
    [
        ("pink", 100, 1, "unknown noise kind"),
        ("white_fm", 3, 1, "n must be"),
        ("white_fm", 100, 0, "m must satisfy"),
        ("white_fm", 100, 60, "m must satisfy"),
    ],
)
def test_edf_validates(kind: str, n: int, m: int, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        equivalent_degrees_of_freedom(kind, n, m)


def test_octave_taus_default_ceiling_is_a_fifth_of_n() -> None:
    assert octave_taus(1000) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert octave_taus(1000, max_m=499) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    with pytest.raises(ConfigurationError, match="n must be"):
        octave_taus(3)


@pytest.mark.parametrize("kind", NOISE_KINDS)
def test_measured_adev_matches_theory_in_slope_and_amplitude(kind: str) -> None:
    level = LEVELS[kind]
    x = synthetic_phase(kind, 20000, tau0_s=1.0, level=level, seed=5)
    table = allan_table(x, 1.0, noise_kind=kind)
    fit = fit_log_slope(table.tau_s, table.adev)
    assert fit.slope == pytest.approx(theory_slope(kind), abs=0.05)
    theory = theoretical_adev(kind, table.tau_s, level=level)
    worst = float(np.max(np.abs(table.adev / theory - 1.0)))
    assert worst < 0.35, f"{kind}: worst relative departure from theory {worst:.4f}"


@pytest.mark.parametrize("kind", NOISE_KINDS)
def test_noise_identifier_recovers_the_generating_process(kind: str) -> None:
    hits = 0
    for seed in range(20):
        x = synthetic_phase(kind, 2000, tau0_s=1.0, level=LEVELS[kind], seed=4000 + seed)
        hits += int(identify_noise(x).kind == kind)
    assert hits >= 18, f"{kind}: identified {hits}/20"


def test_identify_noise_rejects_short_input() -> None:
    with pytest.raises(ConfigurationError, match="at least 8 samples"):
        identify_noise(np.zeros(7))


def test_identify_noise_on_constant_phase_reports_white_pm() -> None:
    result = identify_noise(np.zeros(32))
    assert result.kind == "white_pm"
    assert result.alpha == 2.0


def test_allan_table_interval_brackets_the_point_estimate() -> None:
    x = synthetic_phase("white_fm", 4000, tau0_s=1.0, level=1e-20, seed=8)
    table = allan_table(x, 1.0, noise_kind="white_fm", confidence=0.95)
    assert table.noise_kind_source == "declared"
    assert np.all(table.lower <= table.adev)
    assert np.all(table.adev <= table.upper)
    assert "noise kind white_fm (declared)" in table.as_text()


def test_allan_table_identifies_the_noise_when_not_told() -> None:
    x = synthetic_phase("random_walk_fm", 4000, tau0_s=1.0, level=1e-24, seed=9)
    table = allan_table(x, 1.0)
    assert table.noise_kind_source == "identified"
    assert table.noise_kind == "random_walk_fm"


def test_allan_table_validates() -> None:
    x = synthetic_phase("white_fm", 200, tau0_s=1.0, level=1e-20, seed=1)
    with pytest.raises(ConfigurationError, match="confidence"):
        allan_table(x, 1.0, confidence=1.0)
    with pytest.raises(ConfigurationError, match="unknown noise kind"):
        allan_table(x, 1.0, noise_kind="pink")


def test_fit_log_slope_recovers_an_exact_power_law() -> None:
    tau = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    y = 3.0 * tau**-0.5
    fit = fit_log_slope(tau, y)
    assert fit.slope == pytest.approx(-0.5, abs=1e-12)
    assert np.exp(fit.intercept) == pytest.approx(3.0, rel=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.slope_stderr == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        (np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]), "at least 3 points"),
        (np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0]), "match in length"),
        (np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 3.0]), "strictly positive"),
    ],
)
def test_fit_log_slope_validates(x: np.ndarray, y: np.ndarray, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        fit_log_slope(x, y)
