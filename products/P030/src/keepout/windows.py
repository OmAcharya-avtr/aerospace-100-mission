"""Keep-out-aware pointing windows over an orbit.

A pointing window is a maximal interval of time during which a fixed inertial
target stays clear of every exclusion cone. The cones move: the Earth's centre
tracks the anti-nadir direction and its angular radius follows the altitude,
while the Sun and Moon move slowly against the stars.

The window search is a coarse scan of the worst-case clearance margin followed
by a bracketed root find on each sign change. That is exact to the root-finder's
tolerance for margins that cross zero once per bracket; a violation that opens
and closes entirely between two scan samples is missed, which is why the scan
step is an explicit argument and why the sampling guidance is in the README.

Units: seconds since epoch, metres, radians.

References
----------
D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
    Microcosm Press (2013): Sec. 1.3 (circular orbit period and mean motion),
    Sec. 2.6 / Algorithm 10 (perifocal-to-geocentric-equatorial rotation),
    Algorithms 29 and 31 (Sun and Moon directions).
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 5.2 -- solar and lunar geometry for attitude constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from .bodies import (
    EARTH_MU,
    EARTH_RADIUS_M,
    MOON_RADIUS_M,
    SUN_RADIUS_M,
    angular_radius,
    moon_direction_mod,
    sun_direction_mod,
)
from .cones import ExclusionCone, KeepOutSet, body_exclusion_cone
from .geometry import unit

__all__ = [
    "Window",
    "orbital_period",
    "circular_orbit_positions",
    "OrbitPointingProblem",
    "windows_from_margin",
]


@dataclass(frozen=True)
class Window:
    """A clear pointing interval.

    Attributes
    ----------
    start, end : float
        Interval bounds [s since epoch]. ``start <= end``.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError("window bounds must be finite")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    @property
    def duration(self) -> float:
        """Window length [s]."""
        return float(self.end - self.start)


def orbital_period(
    altitude_m: float, radius_m: float = EARTH_RADIUS_M, mu: float = EARTH_MU
) -> float:
    """Keplerian period of a circular orbit [s].

    ``T = 2 pi sqrt(a^3 / mu)`` with ``a = R + h`` (Vallado 2013, Sec. 1.3).
    Two-body, no oblateness, no drag.

    Parameters
    ----------
    altitude_m : float
        Altitude above the reference sphere [m], ``>= 0``.
    radius_m : float
        Reference body radius [m].
    mu : float
        Gravitational parameter [m^3 s^-2].
    """
    if altitude_m < 0.0:
        raise ValueError("altitude_m must be >= 0")
    a = radius_m + altitude_m
    return float(2.0 * np.pi * np.sqrt(a**3 / mu))


def circular_orbit_positions(
    times_s: ArrayLike,
    altitude_m: float,
    inclination: float,
    raan: float = 0.0,
    arg_lat0: float = 0.0,
    radius_m: float = EARTH_RADIUS_M,
    mu: float = EARTH_MU,
) -> NDArray[np.float64]:
    """Inertial positions on a circular orbit [m].

    With ``a = R + h``, mean motion ``n = sqrt(mu / a^3)`` and argument of
    latitude ``u(t) = u_0 + n t``, the geocentric-equatorial position is the
    perifocal-to-ECI rotation ``R3(-Omega) R1(-i)`` applied to
    ``a (cos u, sin u, 0)`` (Vallado 2013, Algorithm 10, specialised to
    ``e = 0``)::

        x = a (cos O cos u - sin O sin u cos i)
        y = a (sin O cos u + cos O sin u cos i)
        z = a (sin u sin i)

    Parameters
    ----------
    times_s : array_like
        Times since epoch [s].
    altitude_m : float
        Circular altitude [m], ``>= 0``.
    inclination : float
        Inclination [rad], ``0 <= i <= pi``.
    raan : float
        Right ascension of the ascending node [rad].
    arg_lat0 : float
        Argument of latitude at epoch [rad].
    radius_m, mu : float
        Reference radius [m] and gravitational parameter [m^3 s^-2].

    Returns
    -------
    ndarray
        Shape ``(..., 3)``, inertial position [m].

    Notes
    -----
    Validity: unperturbed two-body circular motion. No J2, no drag, no
    third-body terms. The node therefore does not regress, which over days
    matters for beta-angle-driven keep-out; see README limitations.
    """
    if altitude_m < 0.0:
        raise ValueError("altitude_m must be >= 0")
    if not 0.0 <= inclination <= np.pi:
        raise ValueError(f"inclination must lie in [0, pi] rad, got {inclination}")
    t = np.asarray(times_s, dtype=float)
    a = radius_m + altitude_m
    n = np.sqrt(mu / a**3)
    u = arg_lat0 + n * t
    cu, su = np.cos(u), np.sin(u)
    ci, si = np.cos(inclination), np.sin(inclination)
    co, so = np.cos(raan), np.sin(raan)
    return a * np.stack([co * cu - so * su * ci, so * cu + co * su * ci, su * si], axis=-1)


@dataclass(frozen=True)
class OrbitPointingProblem:
    """Sun, Earth and Moon keep-out for an instrument on a circular orbit.

    All exclusion angles are instrument keep-out half-angles [rad], interpreted
    against the body limb or the body centre according to ``reference`` (see
    :func:`keepout.cones.body_exclusion_cone`). A ``None`` angle drops that
    body's cone entirely.

    The Sun and Moon directions come from the low-precision series in
    :mod:`keepout.bodies` and are corrected for the spacecraft's offset from the
    geocentre, which matters for the Moon (about 1 deg from low Earth orbit) and
    not for the Sun (about 0.0024 deg).

    Parameters
    ----------
    epoch_jd : float
        Julian date at ``t = 0`` [days].
    altitude_m : float
        Circular orbit altitude [m].
    inclination : float
        Inclination [rad].
    raan, arg_lat0 : float
        Node and epoch argument of latitude [rad].
    sun_exclusion, earth_exclusion, moon_exclusion : float or None
        Instrument keep-out half-angles [rad].
    reference : {"limb", "center"}
        Convention for the exclusion angles.
    """

    epoch_jd: float
    altitude_m: float
    inclination: float
    raan: float = 0.0
    arg_lat0: float = 0.0
    sun_exclusion: float | None = None
    earth_exclusion: float | None = None
    moon_exclusion: float | None = None
    reference: str = "limb"

    def __post_init__(self) -> None:
        if self.altitude_m < 0.0:
            raise ValueError("altitude_m must be >= 0")
        if self.reference not in ("limb", "center"):
            raise ValueError(f"reference must be 'limb' or 'center', got {self.reference!r}")
        if all(
            x is None for x in (self.sun_exclusion, self.earth_exclusion, self.moon_exclusion)
        ):
            raise ValueError("at least one of the three exclusion angles must be given")

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return orbital_period(self.altitude_m)

    def position(self, t_s: float) -> NDArray[np.float64]:
        """Inertial spacecraft position at ``t_s`` seconds after epoch [m]."""
        return circular_orbit_positions(
            t_s, self.altitude_m, self.inclination, self.raan, self.arg_lat0
        )

    def keepout_at(self, t_s: float) -> KeepOutSet:
        """The keep-out set at ``t_s`` seconds after epoch.

        Returns
        -------
        KeepOutSet
            Cones named ``"sun"``, ``"earth"``, ``"moon"`` for whichever
            exclusion angles were supplied.
        """
        jd = self.epoch_jd + float(t_s) / 86400.0
        r_sc = self.position(float(t_s))
        cones: list[ExclusionCone] = []
        if self.sun_exclusion is not None:
            d_sun, r_sun = sun_direction_mod(jd)
            v = d_sun * float(r_sun) - r_sc
            dist = float(np.linalg.norm(v))
            cones.append(
                body_exclusion_cone(
                    "sun", v, float(angular_radius(SUN_RADIUS_M, dist)),
                    self.sun_exclusion, self.reference,
                )
            )
        if self.earth_exclusion is not None:
            dist = float(np.linalg.norm(r_sc))
            cones.append(
                body_exclusion_cone(
                    "earth", -unit(r_sc), float(angular_radius(EARTH_RADIUS_M, dist)),
                    self.earth_exclusion, self.reference,
                )
            )
        if self.moon_exclusion is not None:
            d_moon, r_moon = moon_direction_mod(jd)
            v = d_moon * float(r_moon) - r_sc
            dist = float(np.linalg.norm(v))
            cones.append(
                body_exclusion_cone(
                    "moon", v, float(angular_radius(MOON_RADIUS_M, dist)),
                    self.moon_exclusion, self.reference,
                )
            )
        return KeepOutSet(tuple(cones))

    def margin(self, t_s: float, target: ArrayLike) -> float:
        """Worst-case clearance of an inertial ``target`` at ``t_s`` [rad].

        Positive means every cone is cleared.
        """
        return float(self.keepout_at(float(t_s)).margin(target))

    def margin_series(self, times_s: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
        """:meth:`margin` evaluated at each of ``times_s`` [rad]."""
        return np.array([self.margin(float(t), target) for t in np.atleast_1d(times_s)])

    def windows(
        self, times_s: ArrayLike, target: ArrayLike, refine: bool = True
    ) -> list[Window]:
        """Pointing windows for a fixed inertial ``target`` over ``times_s``.

        Parameters
        ----------
        times_s : array_like
            Monotonically increasing scan times [s since epoch].
        target : array_like
            Inertial unit vector to the target, shape ``(3,)``.
        refine : bool
            Refine each boundary with Brent's method on the margin function.
            Disable for a cheap scan-resolution answer.

        Returns
        -------
        list of Window
        """
        t = np.asarray(times_s, dtype=float)
        m = self.margin_series(t, target)
        fn = (lambda tt: self.margin(tt, target)) if refine else None
        return windows_from_margin(t, m, fn)


def windows_from_margin(
    times_s: ArrayLike,
    margins: ArrayLike,
    margin_fn=None,
    xtol: float = 1e-6,
) -> list[Window]:
    """Intervals where a sampled margin is non-negative.

    Parameters
    ----------
    times_s : array_like
        Strictly increasing sample times [s], length ``>= 1``.
    margins : array_like
        Margin at each sample [rad]; ``>= 0`` means allowed.
    margin_fn : callable, optional
        ``f(t) -> margin``. When given, each scan-resolution boundary is
        replaced by the Brent root of ``f`` inside the bracketing sample pair.
    xtol : float
        Absolute tolerance on the refined boundary [s].

    Returns
    -------
    list of Window
        Disjoint, in time order. A margin that is non-negative at every sample
        gives one window spanning the whole scan.

    Notes
    -----
    Limitation: this is a sampled search. Any violation that both starts and
    ends between two consecutive samples is invisible to it. Choose a step short
    compared with the fastest cone motion -- for a low Earth orbit the Earth
    cone edge moves at roughly ``360 deg / period``, about 0.066 deg/s at 500 km.
    """
    t = np.asarray(times_s, dtype=float)
    m = np.asarray(margins, dtype=float)
    if t.ndim != 1 or m.ndim != 1 or t.shape != m.shape:
        raise ValueError("times_s and margins must be 1-D arrays of the same length")
    if t.size == 0:
        return []
    if t.size > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    ok = m >= 0.0
    windows: list[Window] = []
    i = 0
    n = t.size
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        start, end = float(t[i]), float(t[j])
        if margin_fn is not None:
            if i > 0:
                start = _root(margin_fn, float(t[i - 1]), float(t[i]), xtol, start)
            if j + 1 < n:
                end = _root(margin_fn, float(t[j]), float(t[j + 1]), xtol, end)
        windows.append(Window(start, end))
        i = j + 1
    return windows


def _root(fn, lo: float, hi: float, xtol: float, fallback: float) -> float:
    """Brent root of ``fn`` on ``[lo, hi]``; ``fallback`` if it does not bracket."""
    try:
        f_lo, f_hi = fn(lo), fn(hi)
    except ValueError:  # pragma: no cover - defensive
        return fallback
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if np.sign(f_lo) == np.sign(f_hi):
        return fallback
    return float(brentq(fn, lo, hi, xtol=xtol))
