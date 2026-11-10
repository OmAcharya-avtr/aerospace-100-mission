"""Ground-station geometry: separation, bearing and the along/cross decomposition.

Every quantity here is a property of the station COORDINATES alone. Nothing in
this module touches a cloud record, and nothing in it is measured: the site set
below is a declared synthetic network, described in ``DATASET_CARD.md``, and the
functions accept arbitrary coordinates so a reader can substitute their own.

Equations
---------
**(1) Great-circle separation.** For stations at geodetic latitude and longitude
``(phi_1, lambda_1)`` and ``(phi_2, lambda_2)`` on a sphere of radius ``R_E``,

    a = sin^2((phi_2 - phi_1) / 2)
        + cos(phi_1) cos(phi_2) sin^2((lambda_2 - lambda_1) / 2)
    d = 2 R_E arcsin(sqrt(a))                                      [km]

the haversine form (Sinnott 1984, *Sky and Telescope* 68(2), 159). Units: input
degrees, output kilometres. Assumption: a spherical Earth of mean radius
``EARTH_RADIUS_KM = 6371.0088`` km, the IUGG mean radius of the WGS-84
ellipsoid. Validity: the spherical approximation misdescribes a geodesic
distance on the ellipsoid by up to about 0.5 per cent (flattening 1/298.257).
That is far below the uncertainty of any correlation length estimated in this
package -- the fitted lengths here carry a sampling uncertainty of order ten per
cent -- so the spherical form is used deliberately and the error is declared
rather than removed.

**(2) Initial bearing.** The azimuth of the great circle at station 1, measured
clockwise from true north,

    theta = atan2( sin(dlambda) cos(phi_2),
                   cos(phi_1) sin(phi_2) - sin(phi_1) cos(phi_2) cos(dlambda) )

in degrees in ``[0, 360)``. Standard spherical trigonometry; see Bowring 1984 or
any navigation text. The bearing is NOT symmetric: the bearing from 1 to 2 and
from 2 to 1 differ by 180 degrees only on a meridian or the equator. The
decomposition of equation (3) uses ``|cos|`` and ``|sin|`` and is therefore
symmetric despite that.

**(3) Along-axis and cross-axis separation.** Given a declared advection axis
``alpha`` (degrees from true north), the separation is resolved as

    d_along = d |cos(theta - alpha)|      d_cross = d |sin(theta - alpha)|  [km]

where ``theta`` is the azimuth of the great circle AT THE MIDPOINT of the arc,
from :func:`midpoint_bearing_deg`, and NOT the initial bearing. The distinction
is not cosmetic and it was found by this product's own property test: built on
the initial bearing, the along and cross matrices of the declared network are
asymmetric by up to 1478.5 km, so the decomposition of a pair depended on which
station had the lower index. The midpoint azimuth is symmetric and the matrices
built on it are symmetric to rounding.

It remains a first-order decomposition. On a sphere the azimuth rotates along
the path, so for separations comparable with the Earth's radius ``d_along`` and
``d_cross`` describe the arc's orientation at its midpoint and are not an
arc-length decomposition of it. The relation ``d_along^2 + d_cross^2 == d^2``
holds exactly in exact arithmetic and to relative rounding in practice.

Why the decomposition exists: a cloud field advected by a prevailing flow is
elongated along it, so two stations separated along the flow stay correlated
further than two stations separated across it. Anisotropy of this kind is
reported for cloud and rain fields; see for example the ITU-R P.1815 and P.618
treatments of site diversity, which parameterise diversity gain by separation
AND by the orientation of the baseline relative to the prevailing storm
direction. This package measures what an ISOTROPIC two-parameter model costs
when the field is anisotropic, which is why both forms are computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: IUGG mean radius of the WGS-84 ellipsoid, km. Used by equation (1).
EARTH_RADIUS_KM: float = 6371.0088

#: Smallest norm of the summed unit vectors at which a great-circle midpoint
#: is considered well defined. Below it the pair is antipodal and the midpoint
#: bearing is ``nan``. DECLARED.
ANTIPODAL_NORM_FLOOR: float = 1e-8

#: Declared advection axis, degrees clockwise from true north. A westerly flow
#: advects cloud toward roughly 065 degrees in the mid-latitudes of the northern
#: hemisphere; the value is DECLARED for the synthetic record of
#: ``sitediverse.field`` and is not a measurement of any real wind climatology.
ADVECTION_AXIS_DEG: float = 65.0


@dataclass(frozen=True)
class Site:
    """One optical ground station, as coordinates and nothing else.

    Parameters
    ----------
    name:
        Short identifier, used as a column label.
    latitude_deg, longitude_deg:
        Geodetic coordinates in degrees, latitude in ``[-90, 90]`` and longitude
        in ``[-180, 180]``.
    altitude_m:
        Station altitude in metres above mean sea level. Carried for
        completeness and used by NO computation in this package: cloud-cover
        statistics here are a declared function of the site index and the epoch,
        not of altitude. It is a field a reader substituting real coordinates
        will have, and leaving it out would invite them to overload another one.
    cluster:
        Index of the declared regional group this station belongs to. Used only
        to construct the declared network and to label figures.
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    cluster: int

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(
                f"latitude_deg must lie in [-90, 90], got {self.latitude_deg!r} "
                f"for site {self.name!r}"
            )
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError(
                f"longitude_deg must lie in [-180, 180], got "
                f"{self.longitude_deg!r} for site {self.name!r}"
            )
        if self.altitude_m < -500.0:
            raise ValueError(
                f"altitude_m must be at or above -500 m, got {self.altitude_m!r} "
                f"for site {self.name!r}"
            )


#: Centres of the five declared regional groups, (latitude, longitude) in
#: degrees. These are DECLARED synthetic coordinates. They are not the
#: coordinates of any real optical ground station, and no station's climatology
#: is used anywhere in this package. See DATASET_CARD.md.
CLUSTER_CENTRES: tuple[tuple[float, float], ...] = (
    (34.0, -8.0),
    (42.5, 6.0),
    (30.0, 22.0),
    (20.0, -30.0),
    (50.5, -62.0),
)

#: Offsets of the four stations within each group, in degrees. Chosen so that
#: within-group separations run from about 99 km to about 300 km, which is the
#: range over which optical site diversity is usually argued about, and so that
#: the baselines within a group point in four different directions, which is
#: what makes the anisotropy of equation (3) observable at all.
CLUSTER_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.6, 0.9),
    (-1.1, 0.4),
    (0.35, -1.7),
)

#: Declared station altitudes in metres, one per offset index. A high site is
#: drier; the value is carried and not used, as ``Site.altitude_m`` says.
CLUSTER_ALTITUDES: tuple[float, float, float, float] = (2400.0, 1150.0, 780.0, 1900.0)


def _declared_network() -> tuple[Site, ...]:
    """Build the declared 20-station network from the constants above."""
    sites: list[Site] = []
    for cluster, (lat, lon) in enumerate(CLUSTER_CENTRES):
        for index, (dlat, dlon) in enumerate(CLUSTER_OFFSETS):
            sites.append(
                Site(
                    name=f"S{cluster + 1}{index + 1}",
                    latitude_deg=lat + dlat,
                    longitude_deg=lon + dlon,
                    altitude_m=CLUSTER_ALTITUDES[index],
                    cluster=cluster,
                )
            )
    return tuple(sites)


#: The declared network: 20 synthetic stations in 5 groups of 4.
SITES: tuple[Site, ...] = _declared_network()

#: Station names in index order.
SITE_NAMES: tuple[str, ...] = tuple(site.name for site in SITES)


def great_circle_km(
    latitude_a_deg: np.ndarray | float,
    longitude_a_deg: np.ndarray | float,
    latitude_b_deg: np.ndarray | float,
    longitude_b_deg: np.ndarray | float,
) -> np.ndarray:
    """Great-circle separation in kilometres, equation (1).

    Parameters
    ----------
    latitude_a_deg, longitude_a_deg, latitude_b_deg, longitude_b_deg:
        Degrees. Broadcast against one another.

    Returns
    -------
    numpy.ndarray
        Separation in kilometres, the same shape as the broadcast inputs.
        Non-negative, and exactly ``0.0`` for coincident coordinates.
    """
    phi_a = np.radians(np.asarray(latitude_a_deg, dtype=float))
    phi_b = np.radians(np.asarray(latitude_b_deg, dtype=float))
    dphi = phi_b - phi_a
    dlam = np.radians(
        np.asarray(longitude_b_deg, dtype=float)
        - np.asarray(longitude_a_deg, dtype=float)
    )
    inner = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi_a) * np.cos(phi_b) * np.sin(dlam / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def initial_bearing_deg(
    latitude_a_deg: np.ndarray | float,
    longitude_a_deg: np.ndarray | float,
    latitude_b_deg: np.ndarray | float,
    longitude_b_deg: np.ndarray | float,
) -> np.ndarray:
    """Initial great-circle bearing from A to B, equation (2).

    Returns
    -------
    numpy.ndarray
        Degrees clockwise from true north, in ``[0, 360)``.
    """
    phi_a = np.radians(np.asarray(latitude_a_deg, dtype=float))
    phi_b = np.radians(np.asarray(latitude_b_deg, dtype=float))
    dlam = np.radians(
        np.asarray(longitude_b_deg, dtype=float)
        - np.asarray(longitude_a_deg, dtype=float)
    )
    east = np.sin(dlam) * np.cos(phi_b)
    north = np.cos(phi_a) * np.sin(phi_b) - np.sin(phi_a) * np.cos(phi_b) * np.cos(dlam)
    # `% 360.0` alone does NOT guarantee the half-open range it looks like it
    # does: for a tiny negative argument the IEEE remainder rounds up to exactly
    # 360.0, which this product's own property test found at an argument of
    # -1e-20. The result is folded again so that 360.0 becomes 0.0.
    wrapped = np.degrees(np.arctan2(east, north)) % 360.0
    return np.where(wrapped >= 360.0, 0.0, wrapped)


def midpoint_bearing_deg(
    latitude_a_deg: np.ndarray | float,
    longitude_a_deg: np.ndarray | float,
    latitude_b_deg: np.ndarray | float,
    longitude_b_deg: np.ndarray | float,
) -> np.ndarray:
    """Azimuth of the great circle at the MIDPOINT of the arc from A to B.

    Why the midpoint and not the departure station. The decomposition of
    equation (3) must be a property of the PAIR, since a pair of stations has
    one separation and one orientation and not two. The initial bearing is not:
    on a sphere the azimuth rotates along the path, so the bearing from A to B
    and the bearing from B to A differ by 180 degrees only on a meridian or the
    equator. Building equation (3) on the initial bearing made the along and
    cross matrices asymmetric by up to 1478.5 km on the declared network, which
    this product's own property test caught; the quantity was then
    ill-defined, because which of two numbers a pair got depended on which
    station happened to have the lower index.

    The azimuth at the arc midpoint is symmetric: the midpoint of the arc from
    A to B is the midpoint of the arc from B to A, and the two azimuths there
    differ by exactly 180 degrees, which equation (3) is invariant under
    because it takes absolute values.

    The midpoint is computed in Cartesian form -- the normalised sum of the two
    unit vectors -- which is exact for any pair except an antipodal one, where
    the sum is the zero vector and the great circle is not unique. An antipodal
    pair returns a bearing of ``nan`` rather than an arbitrary number, and
    ``along_cross_km`` propagates it; no pair of the declared network is
    anywhere near antipodal and the measured minimum norm is reported in
    ``validation/validate_geometry_output.txt``.

    Returns
    -------
    numpy.ndarray
        Degrees clockwise from true north at the midpoint, in ``[0, 360)``, or
        ``nan`` for an antipodal pair.
    """
    lat_a, lon_a, lat_b, lon_b = np.broadcast_arrays(
        np.asarray(latitude_a_deg, dtype=float),
        np.asarray(longitude_a_deg, dtype=float),
        np.asarray(latitude_b_deg, dtype=float),
        np.asarray(longitude_b_deg, dtype=float),
    )
    phi_a = np.radians(lat_a)
    lam_a = np.radians(lon_a)
    phi_b = np.radians(lat_b)
    lam_b = np.radians(lon_b)
    unit_a = np.stack(
        [np.cos(phi_a) * np.cos(lam_a), np.cos(phi_a) * np.sin(lam_a), np.sin(phi_a)],
        axis=-1,
    )
    unit_b = np.stack(
        [np.cos(phi_b) * np.cos(lam_b), np.cos(phi_b) * np.sin(lam_b), np.sin(phi_b)],
        axis=-1,
    )
    total = unit_a + unit_b
    norm = np.linalg.norm(total, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        middle = total / norm[..., None]
    middle = np.where(norm[..., None] > ANTIPODAL_NORM_FLOOR, middle, np.nan)
    mid_lat = np.degrees(np.arcsin(np.clip(middle[..., 2], -1.0, 1.0)))
    mid_lon = np.degrees(np.arctan2(middle[..., 1], middle[..., 0]))
    return initial_bearing_deg(mid_lat, mid_lon, lat_b, lon_b)


def along_cross_km(
    separation_km: np.ndarray | float,
    bearing_deg: np.ndarray | float,
    axis_deg: float = ADVECTION_AXIS_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a separation onto a declared axis and its normal, equation (3).

    Returns
    -------
    tuple of numpy.ndarray
        ``(along_km, cross_km)``, both non-negative, satisfying
        ``along**2 + cross**2 == separation**2`` to within rounding.
    """
    separation = np.asarray(separation_km, dtype=float)
    if np.any(separation < 0.0):
        raise ValueError("separation_km must be non-negative")
    angle = np.radians(np.asarray(bearing_deg, dtype=float) - float(axis_deg))
    return separation * np.abs(np.cos(angle)), separation * np.abs(np.sin(angle))


def separation_matrix(sites: tuple[Site, ...] = SITES) -> np.ndarray:
    """Pairwise great-circle separations, km, shape ``(n, n)``, zero diagonal."""
    lat = np.array([site.latitude_deg for site in sites], dtype=float)
    lon = np.array([site.longitude_deg for site in sites], dtype=float)
    matrix = great_circle_km(lat[:, None], lon[None, :], lat[None, :], lon[:, None])
    np.fill_diagonal(matrix, 0.0)
    return matrix


def midpoint_bearing_matrix(sites: tuple[Site, ...] = SITES) -> np.ndarray:
    """Pairwise midpoint azimuths, degrees, shape ``(n, n)``, zero diagonal.

    Symmetric to within 180 degrees by construction, which is what equation (3)
    needs. The diagonal is a declared zero.
    """
    lat = np.array([site.latitude_deg for site in sites], dtype=float)
    lon = np.array([site.longitude_deg for site in sites], dtype=float)
    matrix = midpoint_bearing_deg(
        lat[:, None], lon[None, :], lat[None, :], lon[:, None]
    )
    np.fill_diagonal(matrix, 0.0)
    return matrix


def bearing_matrix(sites: tuple[Site, ...] = SITES) -> np.ndarray:
    """Pairwise INITIAL bearings, degrees, shape ``(n, n)``.

    Reported by the CLI for information. Equation (3) uses
    ``midpoint_bearing_matrix`` instead, for the reason given there.

    The diagonal is set to ``0.0``: the bearing from a station to itself is
    undefined, and ``arctan2(0, 0)`` returning zero is an artefact rather than
    an answer, so it is overwritten with a declared value.
    """
    lat = np.array([site.latitude_deg for site in sites], dtype=float)
    lon = np.array([site.longitude_deg for site in sites], dtype=float)
    matrix = initial_bearing_deg(
        lat[:, None], lon[None, :], lat[None, :], lon[:, None]
    )
    np.fill_diagonal(matrix, 0.0)
    return matrix


def along_cross_matrices(
    sites: tuple[Site, ...] = SITES, axis_deg: float = ADVECTION_AXIS_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise along-axis and cross-axis separations, km, equation (3).

    Built on the MIDPOINT azimuth, so both matrices are symmetric. The diagonal
    is zero in both.
    """
    along, cross = along_cross_km(
        separation_matrix(sites), midpoint_bearing_matrix(sites), axis_deg
    )
    np.fill_diagonal(along, 0.0)
    np.fill_diagonal(cross, 0.0)
    return along, cross


def upper_triangle_pairs(count: int) -> tuple[tuple[int, int], ...]:
    """Every unordered pair of distinct indices below ``count``, in order."""
    if count < 2:
        raise ValueError(f"count must be at least 2 to form a pair, got {count!r}")
    rows, cols = np.triu_indices(count, 1)
    return tuple((int(i), int(j)) for i, j in zip(rows, cols, strict=True))
