"""Ground stations and their horizon masks.

A station is a geodetic position, an azimuth-dependent minimum-elevation mask,
and a turnaround time between consecutive contacts on the same antenna. All of
these are declared inputs; none is looked up from a catalogue. Real networks
hold this information in vendor-specific systems together with service level
agreements and antenna-by-antenna keyhole and slew constraints that this
package does not model, and that is stated in README.md.

Units: degrees for angles, km for altitude, seconds for times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import geodetic_to_ecef_km, sez_matrix


@dataclass(frozen=True)
class HorizonMask:
    """Minimum usable elevation as a function of azimuth.

    Attributes
    ----------
    floor_deg : float
        Elevation floor applied at every azimuth, degrees. A station with a
        clear horizon and a 5 deg operational limit has ``floor_deg=5.0`` and no
        breakpoints.
    breakpoints : tuple of tuple of float
        ``((azimuth_deg, min_elevation_deg), ...)``, in any order. Azimuths are
        reduced modulo 360 and breakpoints sharing an azimuth are collapsed to
        their maximum elevation, since two elevations at one azimuth do not
        define a function and an obstruction can only raise the usable horizon.
        Between breakpoints the mask is linearly interpolated the short way
        around the circle; with fewer than two *distinct* azimuths only
        ``floor_deg`` applies. The effective mask is the
        pointwise maximum of the floor and the interpolated profile, because an
        obstruction cannot make a station see lower than its operational limit.

    Notes
    -----
    Linear interpolation in azimuth is a modelling choice, not a physical law. A
    real terrain mask is sampled at the station and is usually stepwise; the
    difference matters only for passes that clip a mask edge, and
    ``validation/validate_visibility.py`` reports how many windows in its test
    set have a boundary set by a mask breakpoint rather than by the floor.
    """

    floor_deg: float = 5.0
    breakpoints: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not -90.0 < float(self.floor_deg) < 90.0:
            raise ValueError(f"mask floor must be within (-90, 90) deg, got {self.floor_deg}")
        for azimuth, elevation in self.breakpoints:
            if not np.isfinite(azimuth):
                raise ValueError(f"mask azimuth must be finite, got {azimuth}")
            if not -90.0 < float(elevation) < 90.0:
                raise ValueError(
                    f"mask elevation must be within (-90, 90) deg, got {elevation}"
                )

    def min_elevation_deg(self, azimuth_deg: np.ndarray) -> np.ndarray:
        """Effective minimum elevation at each azimuth, degrees.

        Parameters
        ----------
        azimuth_deg : numpy.ndarray
            Azimuths, degrees, any real values.

        Returns
        -------
        numpy.ndarray
            Same shape as the input.
        """
        azimuth = np.mod(np.asarray(azimuth_deg, dtype=float), 360.0)
        floor = np.full(azimuth.shape, float(self.floor_deg))
        if len(self.breakpoints) < 2:
            return floor
        # Breakpoints at the same azimuth are collapsed to their MAXIMUM, not
        # left duplicated. Two elevations at one azimuth do not define a
        # function, and numpy.interp resolves such a tie by position in its
        # input array, which made the interpolated profile depend on which
        # 360-degree period the query landed in: Hypothesis falsified
        # tests/test_properties.py::TestHorizonMask::test_is_periodic at
        # breakpoints ((1.1932097083332236, 0.0), (1.1932097083332236, 1.0)).
        # The maximum is the physically conservative choice, since an
        # obstruction raises the usable horizon and cannot lower it. The literal
        # case is pinned in tests/test_regressions.py.
        collapsed: dict[float, float] = {}
        for raw_azimuth, raw_elevation in self.breakpoints:
            key = float(raw_azimuth) % 360.0
            value = float(raw_elevation)
            collapsed[key] = max(collapsed.get(key, value), value)
        if len(collapsed) < 2:
            return floor
        points = sorted(collapsed.items())
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        # Wrap one period on each side so interpolation is periodic in azimuth.
        xs_wrapped = np.concatenate([xs - 360.0, xs, xs + 360.0])
        ys_wrapped = np.concatenate([ys, ys, ys])
        profile = np.interp(azimuth, xs_wrapped, ys_wrapped)
        return np.maximum(floor, profile)


@dataclass(frozen=True)
class GroundStation:
    """One antenna at one site.

    Attributes
    ----------
    name : str
        Identifier used in reports and as a dictionary key.
    lat_deg : float
        Geodetic latitude, degrees.
    lon_deg : float
        Longitude east of Greenwich, degrees.
    alt_km : float
        Height above the WGS-84 ellipsoid, km.
    mask : HorizonMask
        Minimum usable elevation profile.
    turnaround_s : float
        Seconds of antenna reconfiguration required between the end of one
        contact and the start of the next on this antenna. Enforced by both
        schedulers as a hard separation; it is why two contacts that merely
        touch at a boundary still conflict.

    Notes
    -----
    One :class:`GroundStation` is one antenna, not one site. A site with three
    dishes is three stations with the same coordinates and separate names, which
    is how the examples model a multi-antenna site.
    """

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    mask: HorizonMask = field(default_factory=HorizonMask)
    turnaround_s: float = 300.0

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("station name must be a non-empty string")
        if float(self.turnaround_s) < 0.0:
            raise ValueError(f"turnaround_s must be non-negative, got {self.turnaround_s}")
        # Validates latitude, longitude and altitude, and fails fast if absurd.
        geodetic_to_ecef_km(self.lat_deg, self.lon_deg, self.alt_km)

    @property
    def ecef_km(self) -> np.ndarray:
        """Earth-fixed position, km, shape ``(3,)``."""
        return geodetic_to_ecef_km(self.lat_deg, self.lon_deg, self.alt_km)

    @property
    def geocentric_radius_km(self) -> float:
        """Distance from the Earth's centre, km."""
        return float(np.linalg.norm(self.ecef_km))

    @property
    def sez(self) -> np.ndarray:
        """Earth-fixed to south-east-zenith rotation, shape ``(3, 3)``."""
        return sez_matrix(self.lat_deg, self.lon_deg)


def default_network() -> tuple[GroundStation, ...]:
    """A four-antenna declared network used by the examples and validation.

    The sites are at coordinates chosen to give a spread of latitudes and
    longitudes and are not real facilities. Two of the four share a site so that
    the multi-antenna case is exercised; one carries a terrain mask with a ridge
    to the north so that mask-limited window boundaries occur.
    """
    ridge = HorizonMask(
        floor_deg=5.0,
        breakpoints=(
            (0.0, 14.0),
            (45.0, 11.0),
            (90.0, 6.0),
            (180.0, 5.0),
            (270.0, 7.0),
            (315.0, 12.0),
        ),
    )
    return (
        GroundStation("SVALBARD-1", 78.23, 15.39, 0.45, HorizonMask(floor_deg=5.0), 300.0),
        GroundStation("SVALBARD-2", 78.23, 15.39, 0.45, HorizonMask(floor_deg=8.0), 300.0),
        GroundStation("PUNTA-ARENAS", -53.16, -70.91, 0.03, ridge, 420.0),
        GroundStation("KOUROU", 5.25, -52.80, 0.01, HorizonMask(floor_deg=7.0), 360.0),
    )
