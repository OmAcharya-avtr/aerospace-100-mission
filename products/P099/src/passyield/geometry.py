"""Ground-station pass geometry from an SGP4 mean-element propagation.

WHAT THIS MODULE IS FOR
-----------------------
Every quantity this product reports is an integral of a data rate over an
interval of a ground-station contact, so the contact's *geometry* -- when the
spacecraft rises above the elevation mask, when it sets, and how far away it is
at every instant between -- is the independent variable of the whole product.
This module produces it, and it produces it with `sgp4` rather than by
hand-rolling a propagator.

THE PROPAGATION MODEL AND ITS LIMITS
------------------------------------
SGP4/SDP4 as implemented by the `sgp4` package.

- Source: Vallado, D. A., Crawford, P., Hujsak, R. and Kelso, T. S. (2006),
  "Revisiting Spacetrack Report #3", AIAA/AAS Astrodynamics Specialist
  Conference, AIAA 2006-6753. That paper is the reference implementation the
  `sgp4` package is a transliteration of, and it supersedes the original
  Hoots & Roehrich (1980) Spacetrack Report #3.
- SGP4 is a **mean-elements** model. A two-line element set is not an
  osculating state vector: it is a set of Brouwer mean elements fitted to
  observations in the specific analytical theory SGP4 inverts, so the elements
  are only meaningful when fed to SGP4 itself. Converting a TLE to an
  osculating Keplerian state and propagating it with a different integrator is
  a category error, and this module does not do it.
- Output frame: **TEME** (True Equator, Mean Equinox) of date, km and km/s.
  Not J2000, not ITRF.
- Validity range: the published accuracy of SGP4 against a TLE is of order
  1 km in position at epoch degrading to tens of kilometres within a few days
  for a low-Earth object, and the element sets themselves are refreshed on
  roughly that cadence. This module propagates no more than the campaign span
  declared in ``data/generate_passes.py`` from a single element set and says so
  in ``DATASET_CARD.md``; that is a deliberate, documented over-extension for
  a reproducible synthetic campaign, not a claim of orbit-determination
  accuracy.
- Deep-space objects (period above 225 minutes) are routed by the library into
  SDP4. The element set used here is near-Earth, so SGP4 proper is exercised.

THE ELEMENT SET
---------------
:data:`TLE_LINE1` / :data:`TLE_LINE2` are catalogue object 06251, "DELTA 1
DEB", epoch 2006-06-25, taken **verbatim from the official SGP4 verification
element set** ``SGP4-VER.TLE``, which the `sgp4` package installs beside its
own source and which originates in Vallado et al. (2006) above. It is a real,
publicly documented element set, not a synthetic one, and
``validation/validate_geometry.py`` checks the two committed lines
character-for-character against the copy inside the installed package rather
than asking the reader to trust this docstring.

It is a 377 km-perigee, 58.06 degree-inclination, 15.56 revolutions-per-day
object -- a moderate-drag low-Earth case, which is the regime a ground-station
contact schedule is actually written for.

TEME TO EARTH-FIXED
-------------------
:func:`gmst_rad` is a direct implementation of the IAU-1982 Greenwich Mean
Sidereal Time polynomial (Vallado 2013, "Fundamentals of Astrodynamics and
Applications", 4th edition, Algorithm 15). Rotating TEME about the pole by
GMST gives a **pseudo-Earth-fixed** frame, not ITRF: polar motion, the
equation of the equinoxes and UT1-UTC are all neglected.

The size of that approximation, and why it is acceptable here: polar motion is
bounded by about 0.3 arcseconds, which displaces the sub-satellite point by
under 10 m; UT1-UTC is bounded by 0.9 s by the leap-second rule, which rotates
the Earth by under 0.004 degrees, about 420 m at the equator. Against a slant
range of order 1000 km, both are below 0.05 degrees of elevation. The
elevation mask in this product is 10 degrees or higher and the rate model's
sensitivity there is finite, so the induced rate error is far below the
acquisition and re-synchronisation effects the product exists to measure.
``validation/validate_geometry.py`` reports the measured elevation
sensitivity rather than leaving this as an argument.

UNITS
-----
Lengths km, angles degrees on every public boundary (radians internally and
only internally), times Julian date split into integer and fractional parts as
the `sgp4` API requires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

#: Line 1 of the element set, verbatim from ``SGP4-VER.TLE`` as installed by
#: the `sgp4` package; see the module docstring for provenance.
TLE_LINE1 = "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985"

#: Line 2 of the element set, same provenance.
TLE_LINE2 = "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774"

#: Earth equatorial radius, km. WGS-84 defining parameter (NIMA TR8350.2,
#: 3rd edition, 1997, amendment 1, 2000).
WGS84_A_KM = 6378.137

#: WGS-84 first eccentricity squared, dimensionless. Derived from the defining
#: flattening 1/298.257223563 as ``e2 = 2f - f**2``.
WGS84_E2 = 6.69437999014e-3

#: Seconds in a Julian day.
SECONDS_PER_DAY = 86400.0


def tle_source_path() -> Path:
    """Path to the verification element-set file inside the installed `sgp4`.

    Returned so that a validation script can compare the two committed TLE
    lines against the installed copy instead of trusting a docstring.
    """
    import sgp4

    return Path(sgp4.__file__).resolve().parent / "SGP4-VER.TLE"


def satellite() -> Satrec:
    """The propagator for the committed element set.

    Returns
    -------
    Satrec
        An `sgp4` ``Satrec`` initialised from :data:`TLE_LINE1` and
        :data:`TLE_LINE2` with the library's default WGS-72 gravity model,
        which is the model SGP4 was fitted in and the only correct choice for
        a TLE.
    """
    return Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)


@dataclass(frozen=True)
class GroundStation:
    """A ground station: geodetic position, elevation mask and link figures.

    Attributes
    ----------
    name:
        Identifier. Used as the grouping key for the held-out-station split.
    latitude_deg:
        Geodetic latitude, degrees, positive north, in ``[-90, 90]``.
    longitude_deg:
        Geodetic longitude, degrees, positive east.
    altitude_m:
        Height above the WGS-84 ellipsoid, metres.
    mask_deg:
        Elevation mask, degrees. The scheduler books no contact below it and
        this product integrates no rate below it. Must be at least 10 degrees
        because the slant-path attenuation model in :mod:`passyield.link` is
        only declared valid there; see that module.
    gain_over_temperature_db:
        Station receive figure of merit G/T, dB/K.
    zenith_attenuation_db:
        One-way clear-air atmospheric attenuation at the zenith, dB, at the
        downlink frequency. Scaled to the slant path in :mod:`passyield.link`.
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    mask_deg: float
    gain_over_temperature_db: float
    zenith_attenuation_db: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(
                f"latitude_deg must lie in [-90, 90] degrees, got {self.latitude_deg!r}"
            )
        if not -180.0 <= self.longitude_deg <= 360.0:
            raise ValueError(
                f"longitude_deg must lie in [-180, 360] degrees, got {self.longitude_deg!r}"
            )
        if self.mask_deg < 10.0 or self.mask_deg >= 90.0:
            raise ValueError(
                "mask_deg must lie in [10, 90) degrees: the slant-path attenuation model "
                f"in passyield.link is declared valid only above 10 degrees, got {self.mask_deg!r}"
            )
        if not math.isfinite(self.gain_over_temperature_db):
            raise ValueError(
                f"gain_over_temperature_db must be finite, got {self.gain_over_temperature_db!r}"
            )
        if self.zenith_attenuation_db < 0.0:
            raise ValueError(
                "zenith_attenuation_db is an attenuation and must be non-negative, got "
                f"{self.zenith_attenuation_db!r}"
            )

    def ecef_km(self) -> np.ndarray:
        """Station position in the pseudo-Earth-fixed frame, km.

        Standard geodetic-to-rectangular conversion on the WGS-84 ellipsoid
        (Vallado 2013, Algorithm 51 site vector; NIMA TR8350.2 for the
        ellipsoid parameters). Units km.
        """
        phi = math.radians(self.latitude_deg)
        lam = math.radians(self.longitude_deg)
        height_km = self.altitude_m / 1000.0
        curvature = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * math.sin(phi) ** 2)
        return np.array(
            [
                (curvature + height_km) * math.cos(phi) * math.cos(lam),
                (curvature + height_km) * math.cos(phi) * math.sin(lam),
                (curvature * (1.0 - WGS84_E2) + height_km) * math.sin(phi),
            ]
        )


def gmst_rad(jd: np.ndarray | float, fr: np.ndarray | float = 0.0) -> np.ndarray:
    """Greenwich Mean Sidereal Time, radians in ``[0, 2*pi)``.

    IAU-1982 GMST polynomial, Vallado 2013 Algorithm 15::

        T   = (JD_UT1 - 2451545.0) / 36525
        s   = 67310.54841 + (876600*3600 + 8640184.812866) T
                          + 0.093104 T**2 - 6.2e-6 T**3      [seconds of time]
        GMST = s / 240                                        [degrees]

    UT1 is approximated by UTC, which is the dominant error of this routine
    and is bounded by 0.9 s of time by the leap-second rule; see the module
    docstring for what that costs in elevation.

    Parameters
    ----------
    jd, fr:
        Julian date, split into a part and a remainder exactly as the `sgp4`
        API splits it. The split is kept because adding the two together in
        double precision loses about 1e-8 days of resolution near JD 2.45e6,
        which is 1 ms -- small, but free to avoid.

    Returns
    -------
    numpy.ndarray
        GMST in radians.
    """
    jd_array = np.asarray(jd, dtype=float)
    fr_array = np.asarray(fr, dtype=float)
    centuries = ((jd_array - 2451545.0) + fr_array) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * centuries
        + 0.093104 * centuries**2
        - 6.2e-6 * centuries**3
    )
    degrees = np.mod(seconds / 240.0, 360.0)
    return np.radians(degrees)


def teme_to_ecef(position_teme_km: np.ndarray, jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Rotate TEME positions into the pseudo-Earth-fixed frame.

    A single rotation about the pole by :func:`gmst_rad`. Polar motion, the
    equation of the equinoxes and UT1-UTC are neglected; the module docstring
    states the resulting bound.

    Parameters
    ----------
    position_teme_km:
        Shape ``(n, 3)``, km.
    jd, fr:
        Shape ``(n,)`` each.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 3)``, km.
    """
    theta = gmst_rad(jd, fr)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x, y, z = position_teme_km[:, 0], position_teme_km[:, 1], position_teme_km[:, 2]
    return np.stack([cos_t * x + sin_t * y, -sin_t * x + cos_t * y, z], axis=1)


def topocentric(
    position_ecef_km: np.ndarray, station: GroundStation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elevation, azimuth and slant range of a target from a station.

    East-north-up rotation of the station-to-target vector (Vallado 2013,
    Algorithm 51, RAZEL, expressed in ENU rather than SEZ; the two differ only
    by the sign convention of the horizontal axes and give identical elevation
    and range).

    Parameters
    ----------
    position_ecef_km:
        Shape ``(n, 3)``, km, pseudo-Earth-fixed.
    station:
        The observing station.

    Returns
    -------
    tuple of numpy.ndarray
        ``(elevation_deg, azimuth_deg, slant_range_km)``, each shape ``(n,)``.
        Elevation is geometric: no refraction correction is applied, which at
        10 degrees is about 0.09 degrees of apparent lift (standard refraction
        tables, e.g. Vallado 2013 section 4.5) and is neglected deliberately
        because the mask here is a scheduling convention rather than a
        measurement of the horizon.
    """
    relative = np.asarray(position_ecef_km, dtype=float) - station.ecef_km()
    phi = math.radians(station.latitude_deg)
    lam = math.radians(station.longitude_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    east = -sin_lam * relative[:, 0] + cos_lam * relative[:, 1]
    north = (
        -sin_phi * cos_lam * relative[:, 0]
        - sin_phi * sin_lam * relative[:, 1]
        + cos_phi * relative[:, 2]
    )
    up = (
        cos_phi * cos_lam * relative[:, 0]
        + cos_phi * sin_lam * relative[:, 1]
        + sin_phi * relative[:, 2]
    )
    slant = np.sqrt(east**2 + north**2 + up**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        elevation = np.degrees(np.arcsin(np.clip(up / slant, -1.0, 1.0)))
    azimuth = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    return elevation, azimuth, slant


def slant_range_spherical_km(elevation_deg: float | np.ndarray, altitude_km: float) -> np.ndarray:
    """Slant range to a circular orbit over a spherical Earth, km.

    The closed-form geometry result

    .. math::
        d(\\varepsilon) = R_e\\left[
            \\sqrt{\\left(\\tfrac{R_e+h}{R_e}\\right)^2 - \\cos^2\\varepsilon}
            - \\sin\\varepsilon\\right]

    with :math:`R_e` the Earth radius and :math:`h` the orbit altitude. Source:
    Maral, G., Bousquet, M. and Sun, Z. (2020), "Satellite Communications
    Systems: Systems, Techniques and Technology", 6th edition, Wiley, chapter
    2 geometry of a satellite link; the same expression appears in ITU-R P.619
    as the slant-path geometry for a spherical Earth.

    Assumptions: spherical Earth of radius :data:`WGS84_A_KM`, circular orbit,
    no refraction. Validity: any elevation in ``[0, 90]`` degrees and any
    positive altitude. Units: km in, km out.

    This function is **not** used by the pass-finding path. It exists as an
    independent closed-form cross-check on the SGP4-plus-rotation chain, and
    ``validation/validate_geometry.py`` reports the measured disagreement, which
    is dominated by the real orbit's eccentricity and the Earth's oblateness --
    both of which the closed form omits by construction.
    """
    if altitude_km <= 0.0:
        raise ValueError(f"altitude_km must be positive, got {altitude_km!r}")
    eps = np.radians(np.asarray(elevation_deg, dtype=float))
    ratio = (WGS84_A_KM + altitude_km) / WGS84_A_KM
    return WGS84_A_KM * (np.sqrt(ratio**2 - np.cos(eps) ** 2) - np.sin(eps))


@dataclass(frozen=True)
class PassWindow:
    """A contact window above a station's elevation mask.

    Attributes
    ----------
    station:
        Station name.
    rise_jd, rise_fr:
        Julian date of mask crossing on the way up, split as `sgp4` splits it.
    duration_s:
        Mask-to-mask duration, seconds. This is the **geometric** window
        duration, which is what a scheduler books and what baseline 1 uses.
    max_elevation_deg:
        Peak geometric elevation during the window, degrees.
    min_slant_range_km:
        Slant range at peak elevation, km.
    """

    station: str
    rise_jd: float
    rise_fr: float
    duration_s: float
    max_elevation_deg: float
    min_slant_range_km: float


def _elevation_scalar(satrec: Satrec, station: GroundStation, jd: float, fr: float) -> float:
    """Geometric elevation, degrees, at one instant. Used by the bisection."""
    error, position, _velocity = satrec.sgp4(jd, fr)
    if error != 0:
        raise ValueError(f"sgp4 propagation failed with error code {error} at jd={jd!r}")
    ecef = teme_to_ecef(np.asarray([position], dtype=float), np.asarray([jd]), np.asarray([fr]))
    elevation, _azimuth, _slant = topocentric(ecef, station)
    return float(elevation[0])


def topocentric_profile(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    offsets_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elevation, azimuth and slant range at offsets from a window's mask rise.

    Parameters
    ----------
    satrec:
        Propagator, from :func:`satellite`.
    station:
        Observing station.
    window:
        The contact window.
    offsets_s:
        Seconds after ``window.rise_jd + window.rise_fr``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(elevation_deg, azimuth_deg, slant_range_km)``.
    """
    offsets = np.asarray(offsets_s, dtype=float)
    jd = np.full(offsets.shape, window.rise_jd)
    fr = window.rise_fr + offsets / SECONDS_PER_DAY
    array = SatrecArray([satrec])
    error, position, _velocity = array.sgp4(jd, fr)
    if np.any(error != 0):
        codes = sorted(set(int(code) for code in np.asarray(error).ravel() if code != 0))
        raise ValueError(f"sgp4 propagation failed with error codes {codes} over the window")
    ecef = teme_to_ecef(np.asarray(position[0], dtype=float), jd, fr)
    return topocentric(ecef, station)


def elevation_profile(
    satrec: Satrec,
    station: GroundStation,
    window: PassWindow,
    offsets_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elevation and slant range at offsets from a window's mask rise.

    A convenience over :func:`topocentric_profile` for the rate model, which
    needs no azimuth.
    """
    elevation, _azimuth, slant = topocentric_profile(satrec, station, window, offsets_s)
    return elevation, slant


def find_passes(
    satrec: Satrec,
    station: GroundStation,
    start_jd: float,
    start_fr: float,
    span_days: float,
    coarse_step_s: float = 10.0,
    refine_iterations: int = 40,
) -> list[PassWindow]:
    """Every mask-to-mask contact window in a span, by coarse scan and bisection.

    A uniform coarse scan locates sign changes of ``elevation - mask``, and each
    crossing is then refined by bisection on the scalar elevation function. The
    coarse step must be short enough that no whole pass fits between two
    samples: for a 15.6 revolutions-per-day object a 10-degree-mask pass lasts
    of order 300 s, so the 10 s default has a margin of about thirty samples and
    ``validation/validate_geometry.py`` measures the shortest pass found and
    reports the margin rather than asserting the default is safe.

    ``refine_iterations`` bisections reduce the bracket from ``coarse_step_s``
    by :math:`2^{-n}`; 40 iterations on a 10 s bracket reaches 9.1e-12 s, far
    below the double-precision resolution of the Julian date, so the practical
    limit is the arithmetic and not the iteration count.

    Returns
    -------
    list of PassWindow
        In chronological order. Windows clipped by the end of the span are
        discarded, because a half-observed pass has no duration a scheduler
        could book.
    """
    if span_days <= 0.0:
        raise ValueError(f"span_days must be positive, got {span_days!r}")
    if coarse_step_s <= 0.0:
        raise ValueError(f"coarse_step_s must be positive, got {coarse_step_s!r}")
    if refine_iterations < 1:
        raise ValueError(f"refine_iterations must be at least 1, got {refine_iterations!r}")

    samples = int(math.ceil(span_days * SECONDS_PER_DAY / coarse_step_s)) + 1
    offsets = np.arange(samples, dtype=float) * coarse_step_s
    jd = np.full(samples, start_jd)
    fr = start_fr + offsets / SECONDS_PER_DAY
    array = SatrecArray([satrec])
    error, position, _velocity = array.sgp4(jd, fr)
    if np.any(error != 0):
        codes = sorted(set(int(code) for code in np.asarray(error).ravel() if code != 0))
        raise ValueError(f"sgp4 propagation failed with error codes {codes} over the scan")
    ecef = teme_to_ecef(np.asarray(position[0], dtype=float), jd, fr)
    elevation, _azimuth, slant = topocentric(ecef, station)

    above = elevation > station.mask_deg
    rises = np.flatnonzero(~above[:-1] & above[1:])
    sets = np.flatnonzero(above[:-1] & ~above[1:])

    windows: list[PassWindow] = []
    for rise_index in rises:
        later = sets[sets > rise_index]
        if later.size == 0:
            continue  # pass clipped by the end of the span
        set_index = int(later[0])
        rise_time = _bisect_crossing(
            satrec, station, start_jd, start_fr, offsets[rise_index], offsets[rise_index + 1],
            refine_iterations,
        )
        set_time = _bisect_crossing(
            satrec, station, start_jd, start_fr, offsets[set_index + 1], offsets[set_index],
            refine_iterations,
        )
        duration = set_time - rise_time
        if duration <= 0.0:
            continue
        interior = slice(int(rise_index), set_index + 2)
        peak = int(np.argmax(elevation[interior])) + int(rise_index)
        # The Julian date is renormalised so that the fractional part stays
        # inside [0, 1). It keeps the `sgp4` jd/fr split meaningful over a
        # campaign of many days, and it is what lets the committed record
        # round-trip the rise instant to full double precision in a short
        # field: with a fractional part that had grown to tens of days, nine
        # significant digits resolved the rise time only to about 9 ms, which
        # moved the mask-crossing elevation by 2e-4 degrees and tripped the
        # clamp guard in passyield.decompose on reconstruction.
        absolute = start_jd + start_fr + rise_time / SECONDS_PER_DAY
        whole = float(np.floor(absolute))
        windows.append(
            PassWindow(
                station=station.name,
                rise_jd=whole,
                rise_fr=float(absolute - whole),
                duration_s=float(duration),
                max_elevation_deg=float(elevation[peak]),
                min_slant_range_km=float(slant[peak]),
            )
        )
    return windows


def _bisect_crossing(
    satrec: Satrec,
    station: GroundStation,
    start_jd: float,
    start_fr: float,
    below_offset_s: float,
    above_offset_s: float,
    iterations: int,
) -> float:
    """Offset, seconds, where elevation crosses the mask between two brackets.

    ``below_offset_s`` is the end of the bracket where elevation is below the
    mask and ``above_offset_s`` the end where it is above; the two may be in
    either temporal order, which is how the same routine serves rise and set.
    """
    low, high = below_offset_s, above_offset_s
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        value = _elevation_scalar(
            satrec, station, start_jd, start_fr + middle / SECONDS_PER_DAY
        )
        if value > station.mask_deg:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def campaign_epoch(satrec: Satrec) -> tuple[float, float]:
    """The element set's own epoch as a split Julian date.

    A campaign anchored on the element-set epoch keeps the propagation span
    symmetric about the fit and is the least dishonest way to use one TLE for
    an extended synthetic campaign.
    """
    return float(satrec.jdsatepoch), float(satrec.jdsatepochF)


def julian_date(year: int, month: int, day: int, hour: int, minute: int, second: float) -> tuple:
    """Split Julian date from a UTC calendar date, delegating to `sgp4`."""
    return jday(year, month, day, hour, minute, second)
