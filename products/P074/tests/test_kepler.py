"""Closed-form two-body propagation and element conversions."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from orbitdrift.constants import WGS72
from orbitdrift.errors import PropagationError
from orbitdrift.kepler import (
    ClassicalElements,
    elements_to_rv,
    mean_to_true_anomaly,
    orbital_period_s,
    propagate_two_body,
    rv_to_elements,
)

MU = WGS72.mu_km3_s2


def test_circular_equatorial_known_answer() -> None:
    """Hand-calculated: a circular orbit of radius r has speed sqrt(mu/r).

    r = 7000 km, mu = 398600.8 km^3/s^2, so mu/r = 56.942971428571425 km^2/s^2
    and v = 7.546056680715526 km/s. The period is
    2 pi sqrt(7000^3 / 398600.8) = 5828.514018806263 s. Both literals are
    written to the precision double arithmetic actually delivers, because a
    rounded literal and a tight tolerance is a test of the rounding.
    """
    r = np.array([7000.0, 0.0, 0.0])
    speed = np.sqrt(MU / 7000.0)
    assert speed == pytest.approx(7.546056680715526, rel=0.0, abs=1e-12)
    v = np.array([0.0, speed, 0.0])
    elements = rv_to_elements(r, v)
    assert elements.semi_major_axis_km == pytest.approx(7000.0, rel=1e-13, abs=0.0)
    assert elements.eccentricity == pytest.approx(0.0, rel=0.0, abs=1e-14)
    assert elements.inclination_rad == pytest.approx(0.0, rel=0.0, abs=1e-15)
    expected_period = 2.0 * np.pi * np.sqrt(7000.0**3 / MU)
    assert expected_period == pytest.approx(5828.514018806263, rel=0.0, abs=1e-9)
    assert orbital_period_s(r, v) == pytest.approx(expected_period, rel=1e-13, abs=0.0)


def test_mean_to_true_anomaly_known_answers() -> None:
    """Three hand-checkable cases of Kepler's equation.

    At e = 0 the three anomalies coincide, so M = nu exactly. At M = 0 and
    M = pi the orbit is at an apse for any eccentricity, so nu is 0 and pi.
    """
    assert mean_to_true_anomaly(1.0, 0.0) == pytest.approx(1.0, rel=1e-14, abs=0.0)
    assert mean_to_true_anomaly(0.0, 0.5) == pytest.approx(0.0, rel=0.0, abs=1e-13)
    assert mean_to_true_anomaly(np.pi, 0.5) == pytest.approx(np.pi, rel=1e-13, abs=0.0)


def test_mean_to_true_anomaly_matches_the_series_for_small_eccentricity() -> None:
    """nu = M + 2 e sin M + (5/4) e^2 sin 2M + O(e^3) (Brouwer and Clemence 1961).

    An independently derived expression rather than a second call to the
    solver: the series comes from expanding the equation of the centre, and at
    e = 1e-3 the neglected third-order term is about 2e-9 radian.
    """
    ecc = 1e-3
    for m in (0.4, 1.3, 2.9, 4.5):
        series = m + 2.0 * ecc * np.sin(m) + 1.25 * ecc**2 * np.sin(2.0 * m)
        assert mean_to_true_anomaly(m, ecc) == pytest.approx(series, rel=0.0, abs=5e-9)


def test_rv_to_elements_rejects_a_hyperbolic_state() -> None:
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 12.0, 0.0])
    with pytest.raises(PropagationError, match="non-negative"):
        rv_to_elements(r, v)


def test_rv_to_elements_rejects_non_finite_and_degenerate_states() -> None:
    with pytest.raises(PropagationError, match="finite"):
        rv_to_elements(np.array([np.nan, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]))
    with pytest.raises(PropagationError, match="zero magnitude"):
        rv_to_elements(np.zeros(3), np.array([0.0, 7.5, 0.0]))
    with pytest.raises(PropagationError, match="angular momentum"):
        rv_to_elements(np.array([7000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_mean_to_true_anomaly_rejects_an_out_of_range_eccentricity() -> None:
    for ecc in (-0.1, 1.0, 1.5):
        with pytest.raises(PropagationError, match=r"not in \[0, 1\)"):
            mean_to_true_anomaly(0.5, ecc)


def test_propagation_returns_to_the_same_state_after_one_period() -> None:
    r = np.array([7000.0, 1200.0, 2000.0])
    v = np.array([-1.0, 6.4, 3.0])
    period = orbital_period_s(r, v)
    positions, velocities = propagate_two_body(r, v, np.array([0.0, period, 2.0 * period]))
    for index in (0, 1, 2):
        assert np.allclose(positions[index], r, rtol=0.0, atol=1e-8)
        assert np.allclose(velocities[index], v, rtol=0.0, atol=1e-11)


def test_closed_form_conserves_energy_and_angular_momentum_exactly() -> None:
    """The conic is fixed once, so the invariants cannot drift with time.

    This is the property that makes this module usable as a reference for a
    numerical integrator: the integrator's invariants drift and this one's do
    not, so a disagreement between them is the integrator's.
    """
    r = np.array([6900.0, 900.0, -1500.0])
    v = np.array([1.2, 6.9, 2.6])
    times = np.linspace(0.0, 5.0 * orbital_period_s(r, v), 37)
    positions, velocities = propagate_two_body(r, v, times)
    energy = 0.5 * np.sum(velocities**2, axis=1) - MU / np.linalg.norm(positions, axis=1)
    momentum = np.linalg.norm(np.cross(positions, velocities), axis=1)
    assert float(np.ptp(energy) / abs(energy[0])) < 1e-14
    assert float(np.ptp(momentum) / momentum[0]) < 1e-14


def test_equatorial_orbit_puts_the_node_at_zero_by_convention() -> None:
    """Documented convention: an equatorial orbit reports raan = 0."""
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.5, 7.4, 0.0])
    elements = rv_to_elements(r, v)
    assert elements.inclination_rad == pytest.approx(0.0, rel=0.0, abs=1e-15)
    assert elements.raan_rad == 0.0


@settings(max_examples=120, deadline=None)
@given(
    a=st.floats(6600.0, 45000.0, allow_nan=False),
    ecc=st.floats(0.0, 0.85, allow_nan=False),
    inc=st.floats(0.0, np.pi, allow_nan=False),
    raan=st.floats(0.0, 2.0 * np.pi, allow_nan=False),
    argp=st.floats(0.0, 2.0 * np.pi, allow_nan=False),
    nu=st.floats(0.0, 2.0 * np.pi, allow_nan=False),
)
def test_element_round_trip_closes(
    a: float, ecc: float, inc: float, raan: float, argp: float, nu: float
) -> None:
    """Elements to state to elements must return the same conic.

    Only the shape parameters and the inclination are compared. The node and
    the argument of perigee are **not**: at zero inclination the node and the
    argument of perigee are individually undefined and only their sum is, and
    at zero eccentricity the argument of perigee and the true anomaly are
    likewise. Asserting those would be asserting a convention, not a property.

    The inclination tolerance is **derived from the conditioning of the
    recovery**, not chosen. Inclination comes back as ``arccos(h_z / |h|)``,
    and ``d(arccos x)/dx = -1/sqrt(1 - x^2) = -1/sin(i)``, so a relative error
    of a few machine epsilons in the ratio becomes an absolute error of
    ``eps / sin(i)`` in the angle. For a nearly equatorial orbit that is large:
    at ``i = 6e-08`` the floor is about 3e-08 radian, which is half the
    inclination itself.

    Hypothesis found this, at ``a = 6600``, ``e = 0``, ``i = 5.960464477539063e-08``
    with every angle zero, against a flat ``abs=1e-9`` tolerance that the
    hand-written cases -- none of which used an inclination below a degree --
    had no reason to question. The counterexample is pinned with literal inputs
    in ``tests/test_regressions.py``, because ``.hypothesis/`` is gitignored and
    a falsifying example found by one run is lost on a fresh clone.
    """
    elements = ClassicalElements(a, ecc, inc, raan, argp, nu)
    r, v = elements_to_rv(elements)
    if np.linalg.norm(r) < 6600.0:
        return
    recovered = rv_to_elements(r, v)
    assert recovered.semi_major_axis_km == pytest.approx(a, rel=1e-9, abs=1e-6)
    assert recovered.eccentricity == pytest.approx(ecc, rel=1e-7, abs=1e-9)
    inclination_floor = 1e-9 + 8.0 * float(np.finfo(float).eps) / max(
        float(np.sin(inc)), float(np.finfo(float).eps)
    )
    assert recovered.inclination_rad == pytest.approx(
        inc, rel=0.0, abs=inclination_floor
    )


@settings(max_examples=120, deadline=None)
@given(
    m=st.floats(-20.0, 20.0, allow_nan=False),
    ecc=st.floats(0.0, 0.95, allow_nan=False),
)
def test_kepler_solution_satisfies_its_own_equation(m: float, ecc: float) -> None:
    """Substituting the answer back into ``M = E - e sin E`` must close.

    This is the residual of the equation that was solved, which is a different
    statement from "the solver agrees with itself": it is the defining
    equation evaluated independently of the iteration that produced the root.
    """
    nu = mean_to_true_anomaly(m, ecc)
    e_anom = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(0.5 * nu), np.sqrt(1.0 + ecc) * np.cos(0.5 * nu)
    )
    residual = (e_anom - ecc * np.sin(e_anom) - (m % (2.0 * np.pi))) % (2.0 * np.pi)
    residual = min(residual, 2.0 * np.pi - residual)
    assert residual == pytest.approx(0.0, rel=0.0, abs=1e-10)
