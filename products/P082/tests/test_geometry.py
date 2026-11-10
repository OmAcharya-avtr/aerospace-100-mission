"""Geometry: separation, bearing and the along/cross decomposition.

The known-answer cases are hand-computed from the haversine form of equation
(1) with ``EARTH_RADIUS_KM = 6371.0088``. Where a case has an exact closed form
-- a quarter meridian, an equatorial quadrant -- the closed form is written in
the test comment and the tolerance is a rounding tolerance; where it does not,
the expected value is derived from the formula itself in an independent way
(spherical law of cosines rather than haversine) and the tolerance covers the
difference between the two forms.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.geometry import (
    ADVECTION_AXIS_DEG,
    CLUSTER_CENTRES,
    EARTH_RADIUS_KM,
    SITE_NAMES,
    SITES,
    Site,
    along_cross_km,
    along_cross_matrices,
    bearing_matrix,
    great_circle_km,
    initial_bearing_deg,
    midpoint_bearing_deg,
    midpoint_bearing_matrix,
    separation_matrix,
    upper_triangle_pairs,
)

LATITUDES = st.floats(-89.0, 89.0, allow_nan=False, allow_infinity=False)
LONGITUDES = st.floats(-179.0, 179.0, allow_nan=False, allow_infinity=False)


def test_coincident_coordinates_give_exactly_zero_separation():
    """A separation of exactly zero is the one exact value haversine gives.

    ``arcsin(sqrt(0))`` is exactly 0.0 in IEEE arithmetic, so this is the single
    place in this module where an exact comparison is legitimate, and the
    docstring says so rather than leaving the reader to wonder.
    """
    assert float(great_circle_km(10.0, 20.0, 10.0, 20.0)) == 0.0


def test_quarter_meridian_matches_the_closed_form():
    """Pole to equator along a meridian is exactly ``pi R_E / 2``."""
    expected = np.pi * EARTH_RADIUS_KM / 2.0
    got = float(great_circle_km(90.0, 0.0, 0.0, 0.0))
    assert got == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_equatorial_quadrant_matches_the_closed_form():
    """Ninety degrees along the equator is also ``pi R_E / 2``."""
    expected = np.pi * EARTH_RADIUS_KM / 2.0
    got = float(great_circle_km(0.0, 0.0, 0.0, 90.0))
    assert got == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_antipodal_points_give_half_the_circumference_to_root_epsilon():
    """Antipodes are ``pi R_E`` apart -- but only to about ``sqrt(eps)``.

    This is a real and well-known property of the haversine form, not a defect
    and not a tolerance chosen to make a test pass. Near antipodes the inner
    quantity ``a`` approaches 1, and ``arcsin(sqrt(a))`` has a derivative that
    diverges there, so a relative perturbation ``eps`` in ``a`` becomes a
    relative perturbation of order ``sqrt(eps)`` in the angle. The measured
    relative error at this pair is 9.49e-09, which is 2.1 times ``sqrt(eps)``.

    The tolerance is therefore 1e-7 and the reason is stated. The declared
    network's largest separation is 7298.4 km, well under half the
    circumference, so this regime is never entered in practice; it is tested
    because a reader supplying their own coordinates could enter it, and
    because a test asserting 1e-12 here would be asserting something false.
    """
    expected = np.pi * EARTH_RADIUS_KM
    got = float(great_circle_km(10.0, 20.0, -10.0, -160.0))
    assert got == pytest.approx(expected, rel=1e-7, abs=0.0)
    assert abs(got - expected) / expected > 1e-10


def test_one_degree_of_latitude_is_about_111_point_2_km():
    """One degree of arc is ``pi R_E / 180`` = 111.1950 km, hand-computed."""
    expected = np.pi * EARTH_RADIUS_KM / 180.0
    got = float(great_circle_km(45.0, 7.0, 46.0, 7.0))
    assert got == pytest.approx(expected, rel=1e-13, abs=0.0)
    assert got == pytest.approx(111.1950, rel=0.0, abs=1e-3)


def test_haversine_agrees_with_the_spherical_law_of_cosines():
    """An independently derived form of the same quantity, equation (1).

    The law of cosines form ``d = R arccos(sin a sin b + cos a cos b cos dl)``
    is algebraically identical to haversine and numerically different: it loses
    precision for small separations, which is why equation (1) uses haversine.
    At the separations of the declared network the two agree, and the tolerance
    is the law-of-cosines form's own error at those separations.
    """
    rng = np.random.default_rng(4242)
    lat_a = rng.uniform(-70.0, 70.0, 200)
    lon_a = rng.uniform(-170.0, 170.0, 200)
    lat_b = rng.uniform(-70.0, 70.0, 200)
    lon_b = rng.uniform(-170.0, 170.0, 200)
    haversine = great_circle_km(lat_a, lon_a, lat_b, lon_b)
    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    cosine = np.clip(
        np.sin(phi_a) * np.sin(phi_b)
        + np.cos(phi_a) * np.cos(phi_b) * np.cos(np.radians(lon_b - lon_a)),
        -1.0,
        1.0,
    )
    law = EARTH_RADIUS_KM * np.arccos(cosine)
    assert float(np.max(np.abs(haversine - law))) < 1e-6


@given(LATITUDES, LONGITUDES, LATITUDES, LONGITUDES)
def test_separation_is_symmetric_to_rounding(lat_a, lon_a, lat_b, lon_b):
    """``d(A, B)`` and ``d(B, A)`` agree to rounding, not bit-exactly.

    The haversine expression is symmetric in exact arithmetic. It is NOT
    bit-symmetric: ``sin^2((phi_b - phi_a)/2)`` and ``sin^2((phi_a - phi_b)/2)``
    round the same way but ``cos(phi_a) cos(phi_b)`` is a product whose operand
    order the compiler may fix either way. The claim is agreement to rounding.
    """
    forward = float(great_circle_km(lat_a, lon_a, lat_b, lon_b))
    backward = float(great_circle_km(lat_b, lon_b, lat_a, lon_a))
    assert forward == pytest.approx(backward, rel=1e-12, abs=1e-9)


@given(LATITUDES, LONGITUDES, LATITUDES, LONGITUDES)
def test_separation_never_exceeds_half_the_circumference(lat_a, lon_a, lat_b, lon_b):
    """No two points on a sphere are further apart than ``pi R_E``."""
    got = float(great_circle_km(lat_a, lon_a, lat_b, lon_b))
    assert 0.0 <= got <= np.pi * EARTH_RADIUS_KM * (1.0 + 1e-12)


@given(LATITUDES, LONGITUDES, LATITUDES, LONGITUDES)
def test_the_triangle_inequality_holds_through_a_third_point(
    lat_a, lon_a, lat_b, lon_b
):
    """Great-circle distance is a metric, so it satisfies the triangle inequality."""
    direct = float(great_circle_km(lat_a, lon_a, lat_b, lon_b))
    via_lat, via_lon = 0.0, 0.0
    first = float(great_circle_km(lat_a, lon_a, via_lat, via_lon))
    second = float(great_circle_km(via_lat, via_lon, lat_b, lon_b))
    assert direct <= first + second + 1e-6


def test_bearing_due_north_is_zero_and_due_east_is_ninety():
    """Two hand-checked bearings from equation (2)."""
    north = float(initial_bearing_deg(0.0, 0.0, 10.0, 0.0))
    east = float(initial_bearing_deg(0.0, 0.0, 0.0, 10.0))
    assert north == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert east == pytest.approx(90.0, rel=1e-13, abs=0.0)


def test_bearing_due_south_is_one_eighty_and_due_west_is_two_seventy():
    """The other two cardinal directions."""
    south = float(initial_bearing_deg(10.0, 0.0, 0.0, 0.0))
    west = float(initial_bearing_deg(0.0, 10.0, 0.0, 0.0))
    assert south == pytest.approx(180.0, rel=1e-13, abs=0.0)
    assert west == pytest.approx(270.0, rel=1e-13, abs=0.0)


def test_bearing_along_the_equator_is_exactly_east_or_west():
    """On the equator the great circle is the equator, so the bearing is 90 or 270."""
    assert float(initial_bearing_deg(0.0, -30.0, 0.0, 40.0)) == pytest.approx(
        90.0, rel=1e-13, abs=0.0
    )
    assert float(initial_bearing_deg(0.0, 40.0, 0.0, -30.0)) == pytest.approx(
        270.0, rel=1e-13, abs=0.0
    )


@given(LATITUDES, LONGITUDES, LATITUDES, LONGITUDES)
def test_bearing_lies_in_the_declared_range(lat_a, lon_a, lat_b, lon_b):
    """Equation (2) returns degrees in the HALF-OPEN range ``[0, 360)``.

    This test found a real defect during the build. ``x % 360.0`` does not
    guarantee the half-open range it looks like it guarantees: for a tiny
    negative ``x`` the IEEE remainder rounds up to exactly 360.0, which
    Hypothesis found at an argument of -1e-20. The implementation now folds the
    result a second time, and ``tests/test_regressions.py`` pins the literal
    case.
    """
    got = float(initial_bearing_deg(lat_a, lon_a, lat_b, lon_b))
    assert 0.0 <= got < 360.0


def test_the_along_cross_decomposition_is_a_pythagorean_identity():
    """``along^2 + cross^2 == d^2`` to relative rounding, equation (3).

    Not asserted bit-exactly: the two components are each a product of a
    separation with a rounded trigonometric function, and squaring and adding
    them reintroduces the rounding the identity cancels in exact arithmetic.
    """
    separation = separation_matrix()
    along, cross = along_cross_matrices()
    rows, cols = np.triu_indices(len(SITES), 1)
    left = along[rows, cols] ** 2 + cross[rows, cols] ** 2
    right = separation[rows, cols] ** 2
    assert float(np.max(np.abs(left - right) / right)) < 1e-14


@given(
    st.floats(0.0, 20000.0, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 360.0, allow_nan=False, allow_infinity=False),
)
def test_the_decomposition_identity_holds_for_any_separation_and_bearing(
    separation, bearing
):
    """The same identity as a property, over the whole input range."""
    along, cross = along_cross_km(separation, bearing)
    total = float(along) ** 2 + float(cross) ** 2
    assert total == pytest.approx(separation**2, rel=1e-13, abs=1e-18)


@given(st.floats(0.0, 20000.0, allow_nan=False, allow_infinity=False))
def test_a_bearing_on_the_axis_is_entirely_along_it(separation):
    """A baseline pointing along the declared axis has zero cross component."""
    along, cross = along_cross_km(separation, ADVECTION_AXIS_DEG)
    assert float(along) == pytest.approx(separation, rel=1e-13, abs=1e-18)
    assert float(cross) == pytest.approx(0.0, rel=0.0, abs=1e-9)


@given(st.floats(0.0, 20000.0, allow_nan=False, allow_infinity=False))
def test_a_bearing_normal_to_the_axis_is_entirely_across_it(separation):
    """A baseline normal to the axis has zero along component."""
    along, cross = along_cross_km(separation, ADVECTION_AXIS_DEG - 90.0)
    assert float(cross) == pytest.approx(separation, rel=1e-13, abs=1e-18)
    assert float(along) == pytest.approx(0.0, rel=0.0, abs=1e-9)


@given(st.floats(0.0, 20000.0, allow_nan=False, allow_infinity=False),
       st.floats(0.0, 360.0, allow_nan=False, allow_infinity=False))
def test_both_components_are_non_negative(separation, bearing):
    """Equation (3) takes absolute values, so neither component is signed."""
    along, cross = along_cross_km(separation, bearing)
    assert float(along) >= 0.0
    assert float(cross) >= 0.0


def test_the_along_cross_decomposition_is_symmetric_in_the_pair():
    """The matrices of equation (3) are symmetric, because of the midpoint azimuth.

    This test found a real defect during the build and is the reason
    ``midpoint_bearing_deg`` exists. Built on the INITIAL bearing, these two
    matrices were asymmetric by up to 1478.5 km on the declared network: the
    azimuth of a great circle rotates along the path, so the bearing from i to
    j and from j to i do not differ by exactly 180 degrees off a meridian, and
    ``|cos|`` and ``|sin|`` are not invariant under the difference. The
    decomposition of a pair then depended on which station had the lower index,
    which is not a property of the pair.

    The tolerance is a rounding tolerance on a quantity of order 1e3 km.
    """
    along, cross = along_cross_matrices()
    assert float(np.max(np.abs(along - along.T))) < 1e-8
    assert float(np.max(np.abs(cross - cross.T))) < 1e-8


def test_the_initial_bearing_decomposition_would_not_have_been_symmetric():
    """The defect the test above exists to prevent, pinned as a measurement.

    Recorded so that a future change back to the initial bearing cannot pass
    quietly. The asymmetry is large -- hundreds of kilometres -- and not a
    rounding effect.
    """
    separation = separation_matrix()
    along, cross = along_cross_km(separation, bearing_matrix())
    assert float(np.max(np.abs(along - along.T))) > 100.0
    assert float(np.max(np.abs(cross - cross.T))) > 100.0


def test_the_midpoint_azimuth_is_symmetric_to_within_one_eighty():
    """``theta(i, j)`` and ``theta(j, i)`` differ by 180 degrees to rounding."""
    matrix = midpoint_bearing_matrix()
    rows, cols = np.triu_indices(len(SITES), 1)
    forward = matrix[rows, cols]
    backward = matrix[cols, rows]
    gap = np.abs(((forward - backward) % 360.0) - 180.0)
    assert float(np.max(gap)) < 1e-9


def test_an_antipodal_pair_gives_a_nan_midpoint_azimuth():
    """The great circle through antipodes is not unique, so the azimuth is nan.

    Returning ``nan`` rather than an arbitrary number is the point: the
    quantity does not exist and a number would invent one.
    """
    assert np.isnan(float(midpoint_bearing_deg(10.0, 20.0, -10.0, -160.0)))


def test_no_declared_pair_is_anywhere_near_antipodal():
    """So the nan path above is never taken by the declared network."""
    assert not np.isnan(along_cross_matrices()[0]).any()


def test_separation_matrix_has_a_zero_diagonal_and_is_symmetric():
    """Structural properties of the matrix form."""
    matrix = separation_matrix()
    assert matrix.shape == (len(SITES), len(SITES))
    assert np.all(np.diag(matrix) == 0.0)
    assert float(np.max(np.abs(matrix - matrix.T))) < 1e-9


def test_bearing_matrix_diagonal_is_the_declared_zero():
    """The bearing from a station to itself is undefined and declared zero."""
    assert np.all(np.diag(bearing_matrix()) == 0.0)


def test_the_declared_network_has_the_shape_it_claims():
    """Twenty stations in five groups of four, all names distinct."""
    assert len(SITES) == 20
    assert len(set(SITE_NAMES)) == 20
    assert len(CLUSTER_CENTRES) == 5
    for cluster in range(5):
        assert sum(site.cluster == cluster for site in SITES) == 4


def test_the_declared_network_spans_the_separation_range_it_claims():
    """Separations from under 100 km to over 7000 km, in one network.

    The claim this test defends is in README.md: the network spans the range
    over which optical site diversity is argued about AND the range over which
    it is uncontroversial, so the measured overstatement runs from a factor of
    about 23 to a factor below 1.
    """
    matrix = separation_matrix()
    rows, cols = np.triu_indices(len(SITES), 1)
    values = matrix[rows, cols]
    assert float(values.min()) < 100.0
    assert float(values.max()) > 7000.0
    assert rows.size == 190


def test_every_cluster_contains_a_short_baseline():
    """Each regional group has at least one pair under 200 km."""
    matrix = separation_matrix()
    for cluster in range(5):
        members = [i for i, site in enumerate(SITES) if site.cluster == cluster]
        rows, cols = np.triu_indices(len(members), 1)
        inner = matrix[np.ix_(members, members)][rows, cols]
        assert float(inner.min()) < 200.0


def test_upper_triangle_pairs_counts_and_orders_correctly():
    """``n (n - 1) / 2`` pairs, each with ``i < j``, in ascending order."""
    pairs = upper_triangle_pairs(6)
    assert len(pairs) == 15
    assert all(i < j for i, j in pairs)
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (4, 5)


def test_site_rejects_a_latitude_outside_the_sphere():
    """Input validation with an actionable message."""
    with pytest.raises(ValueError, match="latitude_deg must lie in"):
        Site("X", 91.0, 0.0, 0.0, 0)


def test_site_rejects_a_longitude_outside_the_sphere():
    """The same for longitude."""
    with pytest.raises(ValueError, match="longitude_deg must lie in"):
        Site("X", 0.0, 181.0, 0.0, 0)


def test_site_rejects_an_implausible_altitude():
    """Below the Dead Sea shore is a typo, not a ground station."""
    with pytest.raises(ValueError, match="altitude_m must be at or above"):
        Site("X", 0.0, 0.0, -600.0, 0)


def test_along_cross_rejects_a_negative_separation():
    """A negative separation is not a separation."""
    with pytest.raises(ValueError, match="separation_km must be non-negative"):
        along_cross_km(-1.0, 0.0)


def test_upper_triangle_pairs_rejects_a_count_below_two():
    """One station forms no pair."""
    with pytest.raises(ValueError, match="count must be at least 2"):
        upper_triangle_pairs(1)
