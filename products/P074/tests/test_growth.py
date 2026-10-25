"""Analytic growth fits, their refusals, and the rank-deficiency guard."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from orbitdrift.errors import InsufficientDataError
from orbitdrift.growth import (
    FORMS,
    best_form,
    fit_all_forms,
    fit_drag_exponential,
    fit_linear,
    fit_power_law,
    fit_quadratic,
)


def test_exact_power_law_is_recovered_to_machine_precision() -> None:
    """Noiseless data: 3.5 * t^1.75 must come back with a zero residual."""
    ages = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 7.0])
    drift = 3.5 * ages**1.75
    fit = fit_power_law(ages, drift)
    assert fit.exponent == pytest.approx(1.75, rel=1e-12, abs=0.0)
    assert fit.amplitude_km == pytest.approx(3.5, rel=1e-12, abs=0.0)
    assert fit.log_residual_std == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert fit.n_points == 6


def test_fixed_exponent_forms_recover_their_own_amplitude() -> None:
    ages = np.array([0.5, 1.0, 2.0, 4.0])
    assert fit_linear(ages, 2.25 * ages).amplitude_km == pytest.approx(2.25, rel=1e-13, abs=0.0)
    assert fit_quadratic(ages, 0.4 * ages**2).amplitude_km == pytest.approx(
        0.4, rel=1e-13, abs=0.0
    )


def test_drag_exponential_recovers_its_timescale_on_noiseless_data() -> None:
    ages = np.linspace(0.25, 7.0, 28)
    drift = 0.8 * np.expm1(ages / 3.0)
    fit = fit_drag_exponential(ages, drift)
    assert fit.timescale_days == pytest.approx(3.0, rel=1e-3, abs=0.0)
    assert fit.amplitude_km == pytest.approx(0.8, rel=1e-3, abs=0.0)
    assert fit.log_residual_std < 1e-6


def test_repeated_ages_are_refused_rather_than_given_a_zero_exponent() -> None:
    """The rank-deficiency regression test, with literal inputs.

    ``numpy.linalg.lstsq`` on a design matrix whose age column is constant
    returns the least-norm solution, which is an exponent of exactly 0.0 with
    a parameter covariance that is singular, and the reported standard error
    comes out as exactly 0.0. That is a confident-looking answer to a question
    the data cannot answer. Hypothesis found this shape of defect in another
    product of this mission by generating repeated abscissae, so the inputs
    are pinned here as an explicit test rather than left to a generator whose
    database is not committed.
    """
    ages = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    drift = np.array([1.0, 1.1, 0.9, 1.2, 0.95])
    with pytest.raises(InsufficientDataError, match="rank-deficient"):
        fit_power_law(ages, drift)


def test_two_points_at_two_ages_are_refused_for_a_free_exponent() -> None:
    """Two points determine a two-parameter fit with zero residual degrees of freedom.

    The refusal is at three points, not two, so that ``log_residual_std`` is
    never a division by zero dressed up as a measurement.
    """
    with pytest.raises(InsufficientDataError, match="at least 3 points"):
        fit_power_law(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_fewer_than_two_usable_points_is_refused() -> None:
    with pytest.raises(InsufficientDataError, match="at least 2 points"):
        fit_linear(np.array([1.0]), np.array([1.0]))
    with pytest.raises(InsufficientDataError, match="at least 2 points"):
        fit_linear(np.array([0.0, -1.0, 2.0]), np.array([1.0, 1.0, 0.0]))


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(InsufficientDataError, match="same shape"):
        fit_linear(np.array([1.0, 2.0]), np.array([1.0]))


def test_zero_and_negative_drift_values_are_dropped_not_clipped() -> None:
    """A zero at zero age is structural; clipping it would bias the exponent.

    The curve is zero at zero age by construction, and a floor would invent a
    data point there. Dropping leaves the fit on the five real points, and the
    recovered exponent must be the exact one.
    """
    ages = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 7.0])
    drift = np.concatenate([[0.0], 2.0 * np.array([0.5, 1.0, 2.0, 4.0, 7.0]) ** 1.5])
    fit = fit_power_law(ages, drift)
    assert fit.n_points == 5
    assert fit.exponent == pytest.approx(1.5, rel=1e-12, abs=0.0)


def test_interval_is_multiplicative_and_widens_with_the_residual_spread() -> None:
    ages = np.array([0.5, 1.0, 2.0, 4.0, 7.0])
    rng = np.random.default_rng(11)
    drift = 2.0 * ages**1.4 * np.exp(rng.normal(0.0, 0.2, ages.size))
    fit = fit_power_law(ages, drift)
    low, high = fit.interval_km(ages, coverage=0.90)
    centre = fit.predict_km(ages)
    assert np.all(low < centre)
    assert np.all(centre < high)
    # The interval is centred multiplicatively, so the ratio to the centre is
    # the same at every age -- which is exactly the limitation the coverage
    # measurement exists to expose.
    ratios = high / centre
    assert float(np.ptp(ratios)) == pytest.approx(0.0, rel=0.0, abs=1e-12)
    wide, _ = fit.interval_km(ages, coverage=0.99)
    assert float(wide[0]) < float(low[0])


def test_interval_rejects_an_out_of_range_coverage() -> None:
    fit = fit_linear(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    for coverage in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match=r"coverage must be in \(0, 1\)"):
            fit.interval_km(np.array([1.0]), coverage=coverage)


def test_fit_all_forms_omits_a_form_it_cannot_fit_rather_than_filling_it() -> None:
    """Two points: the one-parameter forms fit, the free-exponent ones cannot."""
    fits = fit_all_forms(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert set(fits) == {"linear", "quadratic"}
    assert "power" not in fits
    assert "drag_exponential" not in fits


def test_best_form_prefers_the_true_form_and_breaks_ties_toward_the_simpler() -> None:
    ages = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 7.0])
    assert best_form(fit_all_forms(ages, 2.0 * ages)) == "linear"
    assert best_form(fit_all_forms(ages, 2.0 * ages**2)) == "quadratic"
    # Exactly linear data is fitted perfectly by the linear form and by the
    # free power form; the tie goes to the form with fewer parameters, which
    # is the order of FORMS.
    assert list(FORMS) == ["linear", "quadratic", "power", "drag_exponential"]


def test_best_form_refuses_an_empty_set() -> None:
    with pytest.raises(InsufficientDataError, match="no form could be fitted"):
        best_form({})


@settings(max_examples=100, deadline=None)
@given(
    amplitude=st.floats(1e-4, 1e3, allow_nan=False),
    exponent=st.floats(0.2, 3.0, allow_nan=False),
)
def test_noiseless_power_law_is_always_recovered(amplitude: float, exponent: float) -> None:
    """Any noiseless power law must be recovered, for any amplitude and exponent.

    This is the property Hypothesis is for: it explores amplitudes over seven
    decades and exponents from strongly sublinear to cubic, where a
    hand-written test would pick two or three.
    """
    ages = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0])
    drift = amplitude * ages**exponent
    if np.any(drift <= 1e-9) or np.any(~np.isfinite(drift)):
        return
    fit = fit_power_law(ages, drift)
    assert fit.exponent == pytest.approx(exponent, rel=1e-9, abs=1e-9)
    assert fit.amplitude_km == pytest.approx(amplitude, rel=1e-8, abs=0.0)


@settings(max_examples=100, deadline=None)
@given(
    ages=st.lists(st.floats(0.01, 14.0, allow_nan=False), min_size=1, max_size=12),
    drift=st.lists(st.floats(1e-6, 1e5, allow_nan=False), min_size=1, max_size=12),
)
def test_fit_never_returns_a_non_finite_parameter_or_a_fabricated_zero(
    ages: list[float], drift: list[float]
) -> None:
    """Whatever it accepts, it must return finite numbers; otherwise it must raise.

    The specific failure this guards is a fit that succeeds and reports an
    exponent or a standard error of exactly zero, which is what a
    pseudo-inverse of a rank-deficient system produces and which looks like a
    measurement.
    """
    if len(ages) != len(drift):
        return
    t = np.array(ages)
    y = np.array(drift)
    try:
        fit = fit_power_law(t, y)
    except InsufficientDataError:
        return
    assert np.isfinite(fit.exponent)
    assert np.isfinite(fit.amplitude_km)
    assert np.isfinite(fit.log_residual_std)
    assert fit.amplitude_km > 0.0
    if fit.n_points > 2:
        assert np.isfinite(fit.exponent_stderr)
        # Repeated ages are the case that produces an exactly zero standard
        # error, and those are refused above, so a surviving fit with more
        # than two points must have a strictly positive one unless the data
        # lie exactly on the fitted line.
        residual = np.log(y[(t > 0) & (y > 1e-9)]) - np.log(fit.predict_km(t[(t > 0) & (y > 1e-9)]))
        if float(np.max(np.abs(residual))) > 1e-12:
            assert fit.exponent_stderr > 0.0
