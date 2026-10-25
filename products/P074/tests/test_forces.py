"""The force model, checked against the gradient of its own potential."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from orbitdrift import forces
from orbitdrift.constants import BSTAR_REFERENCE_DENSITY_KG_M2_ER, WGS72

MU = WGS72.mu_km3_s2
RE = WGS72.radius_km


def _central_difference_gradient(function, r: np.ndarray, step: float) -> np.ndarray:
    gradient = np.empty(3)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        gradient[axis] = (function(r + offset) - function(r - offset)) / (2.0 * step)
    return gradient


def test_zonal_acceleration_is_minus_the_gradient_of_the_potential() -> None:
    """The analytic gradient against a central finite difference of the potential.

    Two different expressions for the same quantity. The acceleration is coded
    from the differentiated Legendre form; the potential is coded from the
    undifferentiated one. A sign error, a wrong power of r or a wrong Legendre
    derivative shows up here and nowhere else, because every other check in
    this suite uses the acceleration alone.

    The tolerance is derived, not tuned. A central difference of a function of
    magnitude ``|V|`` with step ``h`` carries a roundoff floor of about
    ``eps |V| / h``; with ``|V|`` of order 5.6e4 km^2/s^2, ``h = 1e-2`` km and
    ``eps = 2.2e-16`` that is about 1.2e-9 km/s^2. The comparison is therefore
    scaled to the norm of the acceleration with an absolute floor of 1e-9, so
    a component that happens to be near zero -- the z component on the
    equator, which is 2e-8 against a 7.7e-3 total -- is not held to a
    tolerance the method cannot deliver.
    """
    cases = [
        np.array([7000.0, 1000.0, 2000.0]),
        np.array([0.0, 0.0, 7500.0]),
        np.array([7200.0, 0.0, 0.0]),
        np.array([-4000.0, 5000.0, -3000.0]),
        np.array([20000.0, 10000.0, 5000.0]),
    ]
    for r in cases:
        analytic = forces.zonal_acceleration_km_s2(r, two_body=True)
        numeric = -_central_difference_gradient(forces.zonal_potential, r, 1e-2)
        scale = float(np.linalg.norm(analytic))
        assert float(np.max(np.abs(analytic - numeric))) < 1e-7 * scale + 1e-9


def test_perturbing_zonal_acceleration_is_minus_the_gradient_of_the_perturbing_potential() -> None:
    """The same check with the two-body term removed from both sides.

    Without this the J2 term is 1e-3 of the total and J3 and J4 are 1e-6, so a
    sign error in J3 would hide inside the tolerance of the test above. Here
    the perturbing terms are the whole signal.
    """

    def perturbing_potential(r: np.ndarray) -> float:
        return float(forces.zonal_potential(r) + MU / np.linalg.norm(r))

    for r in (
        np.array([7000.0, 1000.0, 2000.0]),
        np.array([0.0, 0.0, 7500.0]),
        np.array([-4000.0, 5000.0, -3000.0]),
    ):
        analytic = forces.zonal_acceleration_km_s2(r, two_body=False)
        numeric = -_central_difference_gradient(perturbing_potential, r, 1e-1)
        scale = float(np.linalg.norm(analytic))
        assert float(np.max(np.abs(analytic - numeric))) < 5e-5 * scale + 1e-18


def test_j2_acceleration_matches_the_textbook_closed_form() -> None:
    """Against the familiar closed form, written out independently.

    a_J2 = -(3/2) J2 (mu/r^2)(Re/r)^2 [ (1 - 5(z/r)^2) x/r, ..., (3 - 5(z/r)^2) z/r ]
    (Vallado 2013 eq. 8-58). This is a different algebraic arrangement of the
    same term, so agreement confirms the generic Legendre code reduces to it.
    """
    r = np.array([7000.0, 1000.0, 2000.0])
    rn = float(np.linalg.norm(r))
    u = r[2] / rn
    expected = (
        -1.5
        * WGS72.j2
        * (MU / rn**2)
        * (RE / rn) ** 2
        * np.array(
            [
                (1.0 - 5.0 * u**2) * r[0] / rn,
                (1.0 - 5.0 * u**2) * r[1] / rn,
                (3.0 - 5.0 * u**2) * r[2] / rn,
            ]
        )
    )
    only_j2 = WGS72.__class__(
        name="J2 only",
        mu_km3_s2=MU,
        radius_km=RE,
        j2=WGS72.j2,
        j3=0.0,
        j4=0.0,
    )
    measured = forces.zonal_acceleration_km_s2(r, only_j2, two_body=False)
    assert np.allclose(measured, expected, rtol=1e-12, atol=0.0)


def test_atmosphere_is_continuous_inside_a_band_and_hits_the_table_at_each_base() -> None:
    """At each tabulated base altitude the density must be the tabulated value."""
    table = forces._ATMOSPHERE_TABLE
    for base, rho0, _scale in table:
        assert float(forces.atmospheric_density_kg_m3(base)) == pytest.approx(
            rho0, rel=1e-14, abs=0.0
        )


def test_atmosphere_is_monotonically_decreasing_over_the_sampled_range() -> None:
    """Density must fall with height everywhere, including across band joins."""
    altitude = np.linspace(0.0, 1500.0, 15001)
    density = forces.atmospheric_density_kg_m3(altitude)
    assert np.all(np.diff(density) < 0.0)


def test_atmosphere_band_joins_are_discontinuous_and_the_jump_is_bounded() -> None:
    """The published table is 28 independent fits, so the joins do not match.

    This asserts the discontinuity exists rather than pretending it does not:
    a fixed-step Runge-Kutta method loses its formal order at a discontinuity
    in the right-hand side, which is why the step-refinement study is run with
    drag off for the order measurement and with drag on separately.
    """
    worst = 0.0
    for base in forces._ATMOSPHERE_TABLE[1:, 0]:
        below = float(forces.atmospheric_density_kg_m3(base - 1e-9))
        at = float(forces.atmospheric_density_kg_m3(base))
        worst = max(worst, abs(at - below) / below)
    assert worst > 0.0
    assert worst < 0.60


def test_bstar_to_drag_parameter_known_answer() -> None:
    """C_D A/m = 2 B* / rho_0 with rho_0 = 0.157, so B* = 1e-4 gives 1.27389e-3.

    Hand-calculated: 2 * 1e-4 / 0.157 = 1.2738853503184714e-3 m^2/kg.
    """
    assert BSTAR_REFERENCE_DENSITY_KG_M2_ER == 0.157
    assert float(forces.cd_area_over_mass_m2_kg(1e-4)) == pytest.approx(
        1.2738853503184714e-3, rel=1e-15, abs=0.0
    )


def test_negative_bstar_gives_a_negative_drag_parameter_and_an_along_track_thrust() -> None:
    """A negative B* is a legal TLE and is passed through, not clipped.

    Clipping would silently change the dynamics of the object the user
    supplied. The documented consequence is that the drag term becomes a
    thrust, and this test asserts that it does, so the behaviour is pinned.
    """
    coefficient = float(forces.cd_area_over_mass_m2_kg(-1e-4))
    assert coefficient < 0.0
    r = np.array([RE + 350.0, 0.0, 0.0])
    v = np.array([0.0, 7.6, 0.0])
    acceleration = forces.drag_acceleration_km_s2(r, v, 2451545.0, coefficient)
    assert float(np.dot(acceleration, v)) > 0.0


def test_drag_opposes_the_relative_velocity_for_a_positive_bstar() -> None:
    r = np.array([RE + 350.0, 0.0, 0.0])
    v = np.array([0.0, 7.6, 0.1])
    coefficient = float(forces.cd_area_over_mass_m2_kg(2e-4))
    acceleration = forces.drag_acceleration_km_s2(r, v, 2451545.0, coefficient)
    v_relative = v - np.cross(np.array([0.0, 0.0, 7.292115e-05]), r)
    cosine = float(
        np.dot(acceleration, v_relative)
        / (np.linalg.norm(acceleration) * np.linalg.norm(v_relative))
    )
    assert cosine == pytest.approx(-1.0, rel=0.0, abs=1e-12)


def test_drag_magnitude_against_an_independent_si_evaluation() -> None:
    """The drag magnitude computed entirely in SI units, then converted once.

    This test exists because its first version did not do that. It recomputed
    the same km-based expression the implementation uses and compared the two,
    and so it passed while the implementation divided by 1000 where it should
    have multiplied -- a factor of 1e6, which made the drag term invisible.
    Two evaluations of one expression agree whatever the expression is. The
    version below forms the acceleration in m/s^2 from m^2/kg, kg/m^3 and
    m^2/s^2, which are the units the quantities are defined in, and converts
    to km/s^2 exactly once at the end.

    What actually caught the defect was a negative control in
    ``tests/test_divergence.py``: switching the drag term off and requiring
    the divergence curve to change. It did not change, by 2e-7 relative, and
    that is how a term 1e6 too small announces itself.

    The density is taken at the altitude the function itself computes, not at
    a round 400 km. Those differ: the state is built from the **WGS72** radius
    6378.135 km because that is the radius SGP4's elements are referred to,
    while the geodetic height is measured from the **WGS84** ellipsoid whose
    equatorial radius is 6378.137 km, so the height is 399.998 km.
    """
    from orbitdrift.frames import (
        earth_rotation_vector_rad_s,
        geodetic_altitude_km,
        teme_to_ecef,
    )

    r_km = np.array([RE + 400.0, 0.0, 0.0])
    v_km_s = np.array([0.0, 7.67, 0.0])
    coefficient_m2_kg = 1.2738853503184714e-3
    measured_km_s2 = float(
        np.linalg.norm(
            forces.drag_acceleration_km_s2(r_km, v_km_s, 2451545.0, coefficient_m2_kg)
        )
    )

    altitude_km = float(geodetic_altitude_km(teme_to_ecef(r_km, 2451545.0)))
    assert altitude_km == pytest.approx(399.998, rel=0.0, abs=1e-3)
    density_kg_m3 = float(forces.atmospheric_density_kg_m3(altitude_km))
    v_relative_m_s = (
        np.linalg.norm(v_km_s - np.cross(earth_rotation_vector_rad_s(), r_km)) * 1000.0
    )
    expected_m_s2 = 0.5 * coefficient_m2_kg * density_kg_m3 * v_relative_m_s**2
    assert measured_km_s2 == pytest.approx(expected_m_s2 / 1000.0, rel=1e-12, abs=0.0)


def test_drag_acceleration_is_of_the_order_published_for_a_400_km_orbit() -> None:
    """An order-of-magnitude check against a figure from outside this package.

    Vallado 2013 section 8.6.2 gives the drag acceleration on a typical
    low-Earth satellite at 400 km as of order 1e-7 m/s^2 for a
    ``C_D A / m`` of order 1e-2 m^2/kg. At the 1.27e-3 m^2/kg of this test,
    about an eighth of that, the acceleration must therefore land in the
    1e-8 to 1e-7 m/s^2 decade. A factor-of-1000 unit error puts it outside
    that range in either direction, which is the point: this check does not
    share an expression with the implementation at all, only an order of
    magnitude from the literature.
    """
    r_km = np.array([RE + 400.0, 0.0, 0.0])
    v_km_s = np.array([0.0, 7.67, 0.0])
    magnitude_m_s2 = (
        float(
            np.linalg.norm(
                forces.drag_acceleration_km_s2(r_km, v_km_s, 2451545.0, 1.2738853503184714e-3)
            )
        )
        * 1000.0
    )
    assert 1e-8 < magnitude_m_s2 < 1e-6


def test_third_body_acceleration_against_a_cancellation_free_expansion() -> None:
    """The direct-plus-indirect form against its leading series, no cancellation.

    For |r| << |r_b| the point-mass third-body acceleration expands as
    ``(mu_b/|r_b|^3) * (3 (r . rhat_b) rhat_b - r) + O((|r|/|r_b|)^2)``
    (Montenbruck and Gill 2000, section 3.3). That expression has no
    difference of nearly equal large terms in it, so comparing the two is a
    check on the cancellation in the implemented form rather than a
    restatement of it. The leading term is 1e-4 of the retained terms for the
    Sun, so the two agree to about that.
    """
    from orbitdrift.constants import MU_SUN_KM3_S2

    r = np.array([7000.0, 1000.0, 2000.0])
    sun = np.array([1.4e8, 4.0e7, 1.7e7])
    far = np.array([1e30, 1e30, 1e30])  # a body so distant its term vanishes
    measured = forces.third_body_acceleration_km_s2(r, 0.0, (sun, far))
    distance = float(np.linalg.norm(sun))
    unit = sun / distance
    series = (MU_SUN_KM3_S2 / distance**3) * (3.0 * float(np.dot(r, unit)) * unit - r)
    relative = float(np.linalg.norm(measured - series) / np.linalg.norm(series))
    assert relative < 2e-4


def test_third_body_vanishes_at_the_geocentre() -> None:
    """At r = 0 the direct and indirect terms cancel exactly, by construction."""
    acceleration = forces.third_body_acceleration_km_s2(np.zeros(3), 2451545.0)
    assert np.allclose(acceleration, 0.0, rtol=0.0, atol=1e-18)


def test_force_model_label_and_conservative_flag() -> None:
    assert forces.ForceModel(drag=False, third_body=False).is_conservative is True
    assert forces.ForceModel(drag=True, third_body=False).is_conservative is False
    assert forces.ForceModel(drag=False, third_body=True).is_conservative is False
    label = forces.ForceModel().label()
    assert label == "WGS72: two-body + J2-J4 + drag + luni-solar"
    assert forces.ForceModel(zonal=False, drag=False, third_body=False).label() == (
        "WGS72: two-body"
    )


def test_force_model_with_zonal_off_is_pure_two_body() -> None:
    r = np.array([[7000.0, 1000.0, 2000.0]])
    v = np.array([[-1.0, 6.5, 3.0]])
    model = forces.ForceModel(zonal=False, drag=False, third_body=False)
    measured = model.acceleration_km_s2(r, v, 2451545.0, np.array([0.0]))
    expected = -MU * r / np.linalg.norm(r, axis=-1, keepdims=True) ** 3
    assert np.allclose(measured, expected, rtol=1e-15, atol=0.0)


def test_unimplemented_zonal_degree_is_refused() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        forces._legendre_and_derivative(5, np.array([0.1]))


@settings(max_examples=80, deadline=None)
@given(
    x=st.floats(-40000.0, 40000.0, allow_nan=False),
    y=st.floats(-40000.0, 40000.0, allow_nan=False),
    z=st.floats(-40000.0, 40000.0, allow_nan=False),
)
def test_zonal_acceleration_points_inward_and_is_bounded_by_the_two_body_term(
    x: float, y: float, z: float
) -> None:
    """The perturbation can never exceed 2e-3 of the central term for r > Re.

    J2 is 1.08e-3 and the perturbing acceleration scales as J2 (Re/r)^2 times
    the central term, so for r above the Earth radius the ratio is bounded by
    a few times J2. A bound derived from the coefficients rather than from
    running the code.
    """
    r = np.array([x, y, z])
    radius = float(np.linalg.norm(r))
    if radius < 1.05 * RE:
        return
    central = MU / radius**2
    perturbing = float(np.linalg.norm(forces.zonal_acceleration_km_s2(r, two_body=False)))
    assert perturbing < 4.0 * WGS72.j2 * central


@settings(max_examples=80, deadline=None)
@given(altitude=st.floats(0.0, 2000.0, allow_nan=False))
def test_atmospheric_density_is_positive_and_falls_below_the_surface_value(
    altitude: float,
) -> None:
    density = float(forces.atmospheric_density_kg_m3(altitude))
    assert density > 0.0
    assert density <= 1.225
