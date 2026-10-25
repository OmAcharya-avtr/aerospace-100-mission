"""Divergence curves, the window RMS reduction, and the lower-bound inequality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sgp4.api import Satrec

from orbitdrift import divergence
from orbitdrift.errors import PropagationError
from orbitdrift.forces import ForceModel


def test_lower_bound_is_half_the_drift() -> None:
    assert float(divergence.larger_error_lower_bound_km(10.0)) == 5.0
    assert np.allclose(
        divergence.larger_error_lower_bound_km(np.array([2.0, 8.0])), [1.0, 4.0]
    )


def test_the_triangle_inequality_the_lower_bound_rests_on() -> None:
    """``max(|e1|, |e2|) >= D/2`` where ``D`` is the disagreement.

    An independently derived bound checked against a case where truth is
    known: a two-body orbit propagated in closed form is truth, and two
    deliberately wrong propagations stand in for the two models. The point of
    testing it here is that the inequality is easy to write the wrong way
    round, and a comparison of two implementations would not notice.
    """
    from orbitdrift.kepler import ClassicalElements, elements_to_rv, propagate_two_body

    elements = ClassicalElements(7100.0, 0.01, 0.9, 0.3, 0.7, 0.2)
    r0, v0 = elements_to_rv(elements)
    times = np.linspace(0.0, 20000.0, 41)
    truth, _ = propagate_two_body(r0, v0, times)
    # Two wrong models: each starts from a perturbed state, so each has a
    # computable error against truth.
    model_a, _ = propagate_two_body(r0 + np.array([0.4, 0.0, 0.0]), v0, times)
    model_b, _ = propagate_two_body(r0, v0 * 1.00002, times)
    error_a = np.linalg.norm(model_a - truth, axis=1)
    error_b = np.linalg.norm(model_b - truth, axis=1)
    disagreement = np.linalg.norm(model_a - model_b, axis=1)
    bound = divergence.larger_error_lower_bound_km(disagreement)
    assert np.all(np.maximum(error_a, error_b) >= bound - 1e-9)
    # And the bound is not vacuous: somewhere it is a real constraint.
    assert float(bound.max()) > 0.1


def test_the_lower_bound_is_not_a_bound_on_either_model_individually() -> None:
    """A perfect model compared with a bad one gives a large D and zero error.

    Stated as a test because the README makes the claim and a reader has to be
    able to check that the code agrees with it: ``D/2`` bounds the larger of
    the two errors and says nothing about the smaller.
    """
    from orbitdrift.kepler import ClassicalElements, elements_to_rv, propagate_two_body

    elements = ClassicalElements(7100.0, 0.01, 0.9, 0.3, 0.7, 0.2)
    r0, v0 = elements_to_rv(elements)
    times = np.linspace(0.0, 20000.0, 21)
    truth, _ = propagate_two_body(r0, v0, times)
    bad, _ = propagate_two_body(r0 + np.array([5.0, 0.0, 0.0]), v0, times)
    disagreement = np.linalg.norm(truth - bad, axis=1)
    bound = divergence.larger_error_lower_bound_km(disagreement)
    perfect_error = np.zeros_like(disagreement)
    assert float(bound.max()) > 1.0
    assert np.all(perfect_error < bound)


def test_window_rms_of_a_constant_is_that_constant() -> None:
    times = np.linspace(0.0, 1000.0, 101)
    series = np.full_like(times, 3.0)
    out = divergence.window_rms(times, series, 200.0, np.array([300.0, 500.0, 700.0]))
    assert np.allclose(out, 3.0, rtol=1e-14, atol=0.0)


def test_window_rms_of_a_sinusoid_is_its_amplitude_over_root_two() -> None:
    """Hand-calculated: the RMS of ``A sin`` over a whole period is ``A/sqrt(2)``.

    This is why the drift metric is a window RMS rather than an instantaneous
    value: the radial and cross-track components of the separation oscillate
    at the orbital period, and sampling them at an incommensurate interval
    aliases the oscillation into the growth curve.
    """
    period = 5400.0
    times = np.linspace(0.0, 4.0 * period, 40001)
    series = 2.0 * np.sin(2.0 * np.pi * times / period)
    out = divergence.window_rms(times, series, period, np.array([2.0 * period]))
    assert float(out[0]) == pytest.approx(2.0 / np.sqrt(2.0), rel=2e-4, abs=0.0)


def test_window_rms_handles_a_two_dimensional_series() -> None:
    times = np.linspace(0.0, 1000.0, 101)
    series = np.column_stack([np.full(101, 1.0), np.full(101, 2.0), np.full(101, 3.0)])
    out = divergence.window_rms(times, series, 200.0, np.array([400.0, 600.0]))
    assert out.shape == (2, 3)
    assert np.allclose(out, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], rtol=1e-14, atol=0.0)


def test_window_rms_falls_back_to_the_nearest_sample_when_the_window_is_empty() -> None:
    times = np.array([0.0, 10.0, 20.0])
    out = divergence.window_rms(times, np.array([1.0, 2.0, 3.0]), 1e-9, np.array([11.0]))
    assert float(out[0]) == 2.0


def test_sgp4_states_raises_on_an_error_code_rather_than_returning_nan(
    verification_tle_lines: list[str],
) -> None:
    """A failing propagation must raise, naming the code and the time.

    Catalogue 29141 of the published verification set is one of its deliberate
    failure cases: it is a decaying object, and SGP4 returns code 1 and then
    code 6 within a day of its epoch. A silently NaN row would propagate into
    a divergence curve as a NaN drift and from there into a fitted exponent,
    where nothing downstream checks for it.

    The element set is read from the installed ``sgp4`` distribution rather
    than pasted in, so the test cannot drift from the published file.
    """
    for index in range(0, len(verification_tle_lines), 2):
        if int(verification_tle_lines[index][2:7]) == 29141:
            satrec = Satrec.twoline2rv(
                verification_tle_lines[index], verification_tle_lines[index + 1]
            )
            break
    else:  # pragma: no cover - the case is in the published file
        raise AssertionError("catalogue 29141 is not in SGP4-VER.TLE")
    with pytest.raises(PropagationError, match="error code"):
        divergence.sgp4_states(satrec, np.linspace(0.0, 2.0 * 86400.0, 60))


def test_model_divergence_starts_near_zero_and_grows(moderate_drag_satrec: Satrec) -> None:
    """The curve is seeded from the SGP4 state at epoch, so it starts at zero.

    Not exactly zero at the first reported age, because the first reported age
    is half an orbital period in -- a window RMS needs a window. The first
    value must still be far below the last.
    """
    curve = divergence.model_divergence(
        moderate_drag_satrec,
        span_days=1.0,
        steps_per_rev=120,
        report_every_hours=6.0,
        sample_every=4,
    )
    assert curve.source == "sgp4-vs-numerical"
    assert curve.ages_days[0] < 0.05
    assert float(curve.drift_km[0]) < 0.1 * float(curve.drift_km[-1])
    assert np.all(np.diff(curve.drift_km) > 0.0)


def test_ric_components_add_in_quadrature_to_the_total(moderate_drag_satrec: Satrec) -> None:
    """The RMS of the magnitude equals the quadrature sum of the component RMS values.

    An algebraic identity -- the sum of squares commutes with an orthonormal
    basis change and with averaging -- so this is a check that the projection
    and the reduction were applied in a consistent order.
    """
    curve = divergence.model_divergence(
        moderate_drag_satrec,
        span_days=1.0,
        steps_per_rev=120,
        report_every_hours=8.0,
        sample_every=4,
    )
    quadrature = np.sqrt(np.sum(curve.drift_ric_km**2, axis=1))
    assert np.allclose(quadrature, curve.drift_km, rtol=1e-12, atol=0.0)


def test_in_track_dominates_the_divergence(moderate_drag_satrec: Satrec) -> None:
    """The physical expectation: a mean-motion error grows along track.

    Radial and cross-track separations are periodic; only the in-track one
    accumulates. If this ever failed, either the RIC basis or the force model
    would be wrong, and the README's whole explanation of the curve with it.
    """
    curve = divergence.model_divergence(
        moderate_drag_satrec,
        span_days=1.5,
        steps_per_rev=120,
        report_every_hours=12.0,
        sample_every=4,
    )
    radial, in_track, cross = curve.drift_ric_km[-1]
    assert in_track > 5.0 * radial
    assert in_track > 5.0 * cross


def test_curve_interpolation_at_an_intermediate_age(moderate_drag_satrec: Satrec) -> None:
    curve = divergence.model_divergence(
        moderate_drag_satrec, span_days=1.0, steps_per_rev=120, report_every_hours=6.0
    )
    middle = 0.5 * (curve.ages_days[0] + curve.ages_days[-1])
    value = curve.at(middle)
    assert float(curve.drift_km[0]) < value < float(curve.drift_km[-1])


def test_model_divergence_refuses_a_span_too_short_for_the_step(
    moderate_drag_satrec: Satrec,
) -> None:
    with pytest.raises(PropagationError, match="usable steps"):
        divergence.model_divergence(
            moderate_drag_satrec, span_days=1e-6, steps_per_rev=10, sample_every=100
        )


def test_switching_off_drag_changes_the_divergence(moderate_drag_satrec: Satrec) -> None:
    """A negative control on the force model: the terms must matter.

    If switching the drag term off left the curve unchanged, the term would
    not be reaching the integrator, and every statement in README.md about
    which physical differences the divergence is attributable to would be
    false.

    This is the check that found the drag unit defect. With the conversion
    inverted the two curves agreed to 2.09e-07 relative, which is to say the
    drag term was doing nothing, and no test that recomputed the drag
    expression could have noticed.
    """
    common = {
        "span_days": 1.0,
        "steps_per_rev": 120,
        "report_every_hours": 12.0,
        "sample_every": 4,
    }
    with_drag = divergence.model_divergence(
        moderate_drag_satrec, force=ForceModel(drag=True, third_body=False), **common
    )
    without = divergence.model_divergence(
        moderate_drag_satrec, force=ForceModel(drag=False, third_body=False), **common
    )
    relative = abs(float(with_drag.drift_km[-1]) - float(without.drift_km[-1])) / float(
        without.drift_km[-1]
    )
    assert relative > 1e-2


def test_self_consistency_needs_the_same_object_and_an_ordered_pair() -> None:
    early = Satrec.twoline2rv(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    )
    later = Satrec.twoline2rv(
        "1 00005U 58002B   20287.20333880 -.00000016  00000-0 -22483-4 0  9998",
        "2 00005  34.2443 225.5254 1845686 162.2516 205.2356 10.84869164218149",
    )
    other = Satrec.twoline2rv(
        "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
        "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
    )
    with pytest.raises(PropagationError, match="same object"):
        divergence.self_consistency_divergence(early, other)
    with pytest.raises(PropagationError, match="must be later"):
        divergence.self_consistency_divergence(later, early)


def test_self_consistency_of_an_element_set_against_itself_is_zero() -> None:
    """A degenerate but informative control: a zero epoch gap must be refused.

    And with a gap, comparing a set against a copy of itself would give
    exactly zero, which is what makes a non-zero result in the real case
    meaningful. The copy case is covered by the epoch-gap refusal, so this
    test records the refusal and the reason.
    """
    satrec = Satrec.twoline2rv(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    )
    copy = Satrec.twoline2rv(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    )
    with pytest.raises(PropagationError, match="epoch gap is 0.000000"):
        divergence.self_consistency_divergence(satrec, copy)


@settings(max_examples=40, deadline=None)
@given(
    amplitude=st.floats(0.01, 100.0, allow_nan=False),
    offset=st.floats(0.0, 10.0, allow_nan=False),
)
def test_window_rms_is_bounded_by_the_extremes_of_its_input(
    amplitude: float, offset: float
) -> None:
    """A root-mean-square can never leave the range of ``|y|``.

    An independently derived bound: ``min |y| <= rms <= max |y|``.
    """
    times = np.linspace(0.0, 1000.0, 501)
    series = offset + amplitude * np.sin(times / 97.0)
    out = divergence.window_rms(times, series, 300.0, np.array([200.0, 500.0, 800.0]))
    assert np.all(out >= np.min(np.abs(series)) - 1e-12)
    assert np.all(out <= np.max(np.abs(series)) + 1e-12)
