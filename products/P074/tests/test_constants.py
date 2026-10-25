"""The gravity constants must equal the ones SGP4 itself uses.

This is not a tautology. The independent propagator in
:mod:`orbitdrift.forces` is deliberately run on the same constants as SGP4 so
that the divergence it measures is attributable to model structure rather than
to a constants mismatch. If the literals in :mod:`orbitdrift.constants` drifted
away from ``sgp4.earth_gravity.wgs72``, every divergence number in the
repository would acquire a contribution nobody declared, and no other test
would notice.
"""

from __future__ import annotations

import pytest
from sgp4.earth_gravity import wgs72 as sgp4_wgs72

from orbitdrift.constants import WGS72, WGS84


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (WGS72.mu_km3_s2, sgp4_wgs72.mu),
        (WGS72.radius_km, sgp4_wgs72.radiusearthkm),
        (WGS72.j2, sgp4_wgs72.j2),
        (WGS72.j3, sgp4_wgs72.j3),
        (WGS72.j4, sgp4_wgs72.j4),
    ],
)
def test_wgs72_matches_the_installed_sgp4(ours: float, theirs: float) -> None:
    assert ours == theirs


def test_wgs84_is_not_wgs72() -> None:
    """A copy-paste that made the two models identical would be invisible otherwise."""
    assert WGS84.mu_km3_s2 != WGS72.mu_km3_s2
    assert WGS84.radius_km != WGS72.radius_km
    assert WGS84.j2 != WGS72.j2


def test_zonal_coefficients_are_degree_two_to_four() -> None:
    assert [degree for degree, _ in WGS72.zonal_coefficients] == [2, 3, 4]


def test_wgs72_has_no_flattening_and_wgs84_does() -> None:
    """WGS72 here carries no ellipsoid, so a geodetic call on it cannot silently work."""
    assert WGS72.flattening is None
    assert WGS84.flattening is not None
    assert WGS84.flattening == pytest.approx(1.0 / 298.257223563, rel=1e-15, abs=0.0)
