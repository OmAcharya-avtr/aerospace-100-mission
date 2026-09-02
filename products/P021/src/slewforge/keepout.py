"""Keep-out cones and closed-form violation detection along an eigenaxis arc.

An exclusion cone is a spherical cap on the inertial sky: a unit axis ``c``
and a half-angle ``gamma``. A boresight direction ``n`` violates the cone when
``angle(n, c) < gamma``.

The point of this module is the *arc* test rather than the point test. During
an eigenaxis slew the attitude is ``q(psi) = dq(e, psi) ⊗ q0`` and a body-fixed
boresight ``b`` traces the small circle

    n(psi) = R(e, psi) n0,   n0 = R(q0) b,   psi in [0, Delta]

which is a circle of angular radius ``arccos(e · n0)`` about the eigenaxis.
Its inner product with a cone axis is a single sinusoid,

    f(psi) = n(psi) · c = A cos psi + B sin psi + C
    A = n0·c - (e·n0)(e·c),   B = (e × n0)·c,   C = (e·n0)(e·c)

which follows from Rodrigues' formula by collecting terms. Violation is
``f(psi) > cos gamma``, so every entry and exit is a root of
``R cos(psi - phi) = cos gamma - C`` with ``R = hypot(A, B)`` and
``phi = atan2(B, A)`` -- solved in closed form, no sampling. The violating set
is the single arc ``|psi - phi| < arccos((cos gamma - C)/R)`` modulo ``2 pi``,
and the extreme approach to the cone is at ``psi = phi`` (deepest) and
``psi = phi + pi`` (furthest).

This is the two-cone (small-circle against cap) intersection problem of
spherical geometry; see Wertz (1978) Sec. 11.2 on cone intersections in
attitude geometry.

**Independence.** This implementation shares no code with the sibling product
KeepOut (P030) and imports nothing from it. `validation/validate_keepout_cross_check.py`
compares the two on identical geometry precisely because they were written
separately.

Angles are radians. Directions are Cartesian unit vectors in one common
inertial frame chosen by the caller.

References
----------
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 5.2, 11.2 -- cone geometry, sensor exclusion, cone
    intersections.
D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
    Microcosm Press (2013), Sec. 11.7 -- apparent angular size of a body.
I. Todhunter, *Spherical Trigonometry*, 5th ed., Macmillan (1886) -- the
    spherical relations behind the sinusoid form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attitude import cross3, rotate_about_axis, unit_vector

__all__ = [
    "ASTRONOMICAL_UNIT_M",
    "ArcViolation",
    "EARTH_RADIUS_M",
    "KeepOutSet",
    "KeepOutCone",
    "MOON_RADIUS_M",
    "SUN_RADIUS_M",
    "angular_radius",
    "arc_coefficients_raw",
    "body_keepout_cone",
    "earth_angular_radius",
    "min_margin_on_arc_raw",
]

EARTH_RADIUS_M = 6378137.0
"""WGS-84 equatorial radius [m] (NIMA TR8350.2, 3rd ed., 1997)."""

MOON_RADIUS_M = 1737400.0
"""IAU/IAG mean lunar radius [m] (Archinal et al. 2018)."""

SUN_RADIUS_M = 6.957e8
"""IAU 2015 Resolution B3 nominal solar radius [m]."""

ASTRONOMICAL_UNIT_M = 149597870700.0
"""Astronomical unit [m], IAU 2012 Resolution B2 (exact by definition)."""

_ANG_EPS = 1e-14


def arc_coefficients_raw(
    n0: NDArray[np.float64], eigenaxis: NDArray[np.float64], cone_axis: NDArray[np.float64]
) -> tuple[float, float, float]:
    """``(A, B, C)`` of ``n(psi) · c = A cos psi + B sin psi + C``, unchecked.

    All three arguments must already be unit ``(3,)`` float arrays; nothing is
    validated. This is the inner loop of the planner, called tens of thousands
    of times per solve. :meth:`KeepOutCone.arc_coefficients` is the checked
    entry point and returns identical values --
    ``tests/test_keepout.py::test_raw_and_checked_arc_agree`` asserts they are
    bit-identical over random geometries.
    """
    en = n0[0] * eigenaxis[0] + n0[1] * eigenaxis[1] + n0[2] * eigenaxis[2]
    ec = cone_axis[0] * eigenaxis[0] + cone_axis[1] * eigenaxis[1] + cone_axis[2] * eigenaxis[2]
    nc = n0[0] * cone_axis[0] + n0[1] * cone_axis[1] + n0[2] * cone_axis[2]
    cx = eigenaxis[1] * n0[2] - eigenaxis[2] * n0[1]
    cy = eigenaxis[2] * n0[0] - eigenaxis[0] * n0[2]
    cz = eigenaxis[0] * n0[1] - eigenaxis[1] * n0[0]
    b = cx * cone_axis[0] + cy * cone_axis[1] + cz * cone_axis[2]
    return float(nc - en * ec), float(b), float(en * ec)


def min_margin_on_arc_raw(
    n0: NDArray[np.float64],
    eigenaxis: NDArray[np.float64],
    cone_axis: NDArray[np.float64],
    half_angle: float,
    sweep: float,
) -> float:
    """Closed-form smallest clearance [rad] on ``psi in [0, sweep]``, unchecked.

    Same value as :meth:`KeepOutCone.min_margin_on_arc`; see that method for
    the derivation. Inputs must be unit ``(3,)`` float arrays and
    ``sweep >= 0``.
    """
    a, b, c = arc_coefficients_raw(n0, eigenaxis, cone_axis)
    r = math.hypot(a, b)
    fmax = max(a + c, a * math.cos(sweep) + b * math.sin(sweep) + c)
    if r > _ANG_EPS:
        phi = math.atan2(b, a)
        if 0.0 <= phi <= sweep or 0.0 <= phi + 2.0 * math.pi <= sweep:
            fmax = max(fmax, r + c)
    if fmax > 1.0:
        fmax = 1.0
    elif fmax < -1.0:
        fmax = -1.0
    return math.acos(fmax) - half_angle


def angular_radius(body_radius_m: float, distance_m: float) -> float:
    """Apparent angular radius ``arcsin(R/d)`` [rad] of a sphere.

    Assumptions: spherical body, point observer, no refraction. Valid for
    ``d >= R``; the observer inside the body has no angular radius and raises.
    """
    r = float(body_radius_m)
    d = float(distance_m)
    if r < 0.0:
        raise ValueError(f"body_radius_m must be >= 0, got {r}")
    if d <= 0.0:
        raise ValueError(f"distance_m must be > 0, got {d}")
    if d < r:
        raise ValueError(f"distance_m {d} is inside the body radius {r}; angular radius undefined")
    return float(np.arcsin(r / d))


def earth_angular_radius(altitude_m: float, radius_m: float = EARTH_RADIUS_M) -> float:
    """Angular radius of the Earth [rad] seen from ``altitude_m`` above it.

    ``arcsin(R_E / (R_E + h))``. Uses the WGS-84 *equatorial* radius by
    default, which makes the result an upper bound on the true limb.
    """
    h = float(altitude_m)
    if h < 0.0:
        raise ValueError(f"altitude_m must be >= 0, got {h}")
    return angular_radius(radius_m, radius_m + h)


@dataclass(frozen=True)
class KeepOutCone:
    """A forbidden cap of inertial directions.

    Parameters
    ----------
    axis : array_like
        Cone axis, shape ``(3,)``, normalised on construction. Points at the
        centre of the excluded region, e.g. towards the Sun.
    half_angle : float
        Half-angle [rad] in ``[0, pi]``. A boresight closer to ``axis`` than
        this violates the cone.
    name : str
        Label used in violation reports.

    Notes
    -----
    Cone axes are treated as **static for the duration of one slew**. A slew
    lasting minutes moves the Sun by tenths of an arcminute and the Earth
    nadir by degrees; see README limitations.
    """

    axis: NDArray[np.float64]
    half_angle: float
    name: str = "cone"

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", unit_vector(self.axis).reshape(3))
        ha = float(self.half_angle)
        if not math.isfinite(ha):
            raise ValueError("half_angle must be finite")
        if not 0.0 <= ha <= math.pi:
            raise ValueError(f"half_angle must lie in [0, pi] rad, got {ha}")
        object.__setattr__(self, "half_angle", ha)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    @property
    def half_angle_deg(self) -> float:
        """Half-angle [deg]."""
        return math.degrees(self.half_angle)

    @property
    def solid_angle(self) -> float:
        """Excluded solid angle [sr], ``2 pi (1 - cos(half_angle))``."""
        return float(2.0 * math.pi * (1.0 - math.cos(self.half_angle)))

    def separation(self, direction: ArrayLike) -> NDArray[np.float64] | float:
        """Angle between ``direction`` and the cone axis [rad], in ``[0, pi]``.

        ``atan2(|n × c|, n · c)``; the arccos form loses half its digits for
        nearly parallel vectors.
        """
        n = unit_vector(direction)
        if n.shape == (3,):
            c = cross3(n, self.axis)
            return float(np.arctan2(math.sqrt(float(c @ c)), float(n @ self.axis)))
        cross = np.linalg.norm(np.cross(n, self.axis), axis=-1)
        dot = np.sum(n * self.axis, axis=-1)
        out = np.arctan2(cross, dot)
        return float(out) if np.ndim(out) == 0 else out

    def margin(self, direction: ArrayLike) -> NDArray[np.float64] | float:
        """Clearance [rad]: ``separation - half_angle``. Positive is clear."""
        return self.separation(direction) - self.half_angle

    def contains(self, direction: ArrayLike) -> NDArray[np.bool_] | bool:
        """``True`` strictly inside the cone; the boundary counts as allowed."""
        out = np.asarray(self.margin(direction)) < 0.0
        return bool(out) if np.ndim(out) == 0 else out

    def rotated(self, rotation: ArrayLike) -> KeepOutCone:
        """Same cone with its axis rotated by a ``(3, 3)`` matrix."""
        r = np.asarray(rotation, dtype=float)
        if r.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {r.shape}")
        return replace(self, axis=r @ self.axis)

    # -- the closed-form arc test ---------------------------------------

    def arc_coefficients(self, n0: ArrayLike, eigenaxis: ArrayLike) -> tuple[float, float, float]:
        """Coefficients ``(A, B, C)`` of ``n(psi) · axis = A cos + B sin + C``.

        Parameters
        ----------
        n0 : array_like
            Boresight direction at ``psi = 0``, shape ``(3,)``, inertial.
        eigenaxis : array_like
            Unit rotation axis, shape ``(3,)``, inertial.

        Returns
        -------
        (float, float, float)
            Dimensionless. Derived by substituting Rodrigues' formula for
            ``n(psi)`` into ``n · c`` and collecting ``cos psi``, ``sin psi``
            and the constant.
        """
        n = unit_vector(n0).reshape(3)
        e = unit_vector(eigenaxis).reshape(3)
        return arc_coefficients_raw(n, e, self.axis)

    def min_margin_on_arc(self, n0: ArrayLike, eigenaxis: ArrayLike, sweep: float) -> float:
        """Smallest clearance [rad] over ``psi in [0, sweep]``, in closed form.

        The minimum of ``arccos(f(psi)) - gamma`` is at the maximum of ``f``,
        which for a single sinusoid is either the interior stationary point
        ``psi = phi mod 2 pi`` if it falls inside the interval, or an endpoint.
        No sampling, so a violation narrower than any step size cannot be
        missed.

        Parameters
        ----------
        n0, eigenaxis : array_like
            Shape ``(3,)``, inertial.
        sweep : float
            Total swept angle [rad], ``>= 0``.

        Returns
        -------
        float
            Signed clearance [rad]; negative means the arc enters the cone.
        """
        sw = float(sweep)
        if sw < 0.0:
            raise ValueError(f"sweep must be >= 0 rad, got {sw}")
        n = unit_vector(n0).reshape(3)
        e = unit_vector(eigenaxis).reshape(3)
        return min_margin_on_arc_raw(n, e, self.axis, self.half_angle, sw)

    def violation_intervals(
        self, n0: ArrayLike, eigenaxis: ArrayLike, sweep: float
    ) -> tuple[tuple[float, float], ...]:
        """Closed-form ``psi`` intervals [rad] inside ``[0, sweep]`` that violate.

        Returns an empty tuple when the arc never enters the cone. At most two
        disjoint intervals can be returned for a sweep of ``2 pi`` or less,
        because the violating set is a single arc modulo ``2 pi``.
        """
        sw = float(sweep)
        if sw < 0.0:
            raise ValueError(f"sweep must be >= 0 rad, got {sw}")
        a, b, c = self.arc_coefficients(n0, eigenaxis)
        r = math.hypot(a, b)
        target = math.cos(self.half_angle)
        if r <= _ANG_EPS:
            # f is constant: either the whole arc violates or none of it does.
            return ((0.0, sw),) if c > target else ()
        d = (target - c) / r
        if d >= 1.0:
            return ()  # f never reaches the boundary; max f = r + c <= target
        if d <= -1.0:
            return ((0.0, sw),)  # min f = c - r > target: always violating
        phi = math.atan2(b, a)
        delta = math.acos(d)
        out: list[tuple[float, float]] = []
        kmin = int(math.floor((0.0 - phi - delta) / (2.0 * math.pi))) - 1
        kmax = int(math.ceil((sw - phi + delta) / (2.0 * math.pi))) + 1
        for k in range(kmin, kmax + 1):
            lo = phi - delta + 2.0 * math.pi * k
            hi = phi + delta + 2.0 * math.pi * k
            lo_c, hi_c = max(lo, 0.0), min(hi, sw)
            if hi_c > lo_c:
                out.append((lo_c, hi_c))
        out.sort()
        return tuple(out)


def body_keepout_cone(
    name: str,
    body_direction: ArrayLike,
    body_angular_radius: float,
    exclusion_half_angle: float,
    reference: str = "limb",
) -> KeepOutCone:
    """Build a cone from an instrument keep-out specification.

    Datasheets quote the angle two ways and the difference is the body's own
    angular radius -- about 67 deg for the Earth at 550 km, which no margin
    absorbs.

    - ``reference="limb"``: the angle is measured to the nearest point of the
      body's limb, so the cone half-angle is
      ``body_angular_radius + exclusion_half_angle``.
    - ``reference="center"``: the angle is already measured to the body centre
      and is used unchanged.

    The result is clamped to ``pi`` (a cone covering the whole sky).
    """
    if body_angular_radius < 0.0:
        raise ValueError(f"body_angular_radius must be >= 0 rad, got {body_angular_radius}")
    if exclusion_half_angle < 0.0:
        raise ValueError(f"exclusion_half_angle must be >= 0 rad, got {exclusion_half_angle}")
    if reference == "limb":
        total = float(body_angular_radius) + float(exclusion_half_angle)
    elif reference == "center":
        total = float(exclusion_half_angle)
    else:
        raise ValueError(f"reference must be 'limb' or 'center', got {reference!r}")
    return KeepOutCone(axis=body_direction, half_angle=min(total, math.pi), name=name)


@dataclass(frozen=True)
class ArcViolation:
    """One violating stretch of an eigenaxis arc.

    Attributes
    ----------
    cone : str
        Name of the cone violated.
    boresight : str
        Name of the instrument boresight that violated it.
    psi_start, psi_end : float
        Swept-angle bounds of the violation [rad], within ``[0, sweep]``.
    depth : float
        Worst penetration [rad]: ``half_angle - min separation`` over the
        interval, always ``> 0``.
    """

    cone: str
    boresight: str
    psi_start: float
    psi_end: float
    depth: float

    @property
    def width(self) -> float:
        """Angular width of the violating stretch [rad]."""
        return self.psi_end - self.psi_start


@dataclass(frozen=True)
class KeepOutSet:
    """A set of cones tested together.

    Parameters
    ----------
    cones : sequence of KeepOutCone
    """

    cones: tuple[KeepOutCone, ...] = ()

    def __post_init__(self) -> None:
        cones = tuple(self.cones)
        for c in cones:
            if not isinstance(c, KeepOutCone):
                raise TypeError(f"every element must be a KeepOutCone, got {type(c).__name__}")
        object.__setattr__(self, "cones", cones)

    def __len__(self) -> int:
        return len(self.cones)

    def __iter__(self):
        return iter(self.cones)

    def __getitem__(self, i: int) -> KeepOutCone:
        return self.cones[i]

    @property
    def names(self) -> tuple[str, ...]:
        """Cone names, in order."""
        return tuple(c.name for c in self.cones)

    def with_cone(self, cone: KeepOutCone) -> KeepOutSet:
        """Return a new set with ``cone`` appended."""
        return KeepOutSet(self.cones + (cone,))

    def margins(self, direction: ArrayLike) -> NDArray[np.float64]:
        """Per-cone clearance [rad], shape ``(..., n_cones)``."""
        n = unit_vector(direction)
        if not self.cones:
            return np.empty(n.shape[:-1] + (0,), dtype=float)
        return np.stack([np.asarray(c.margin(n), dtype=float) for c in self.cones], axis=-1)

    def margin(self, direction: ArrayLike) -> NDArray[np.float64] | float:
        """Worst-case clearance [rad]; ``+inf`` for an empty set."""
        m = self.margins(direction)
        if m.shape[-1] == 0:
            return float("inf") if m.ndim == 1 else np.full(m.shape[:-1], np.inf)
        out = np.min(m, axis=-1)
        return float(out) if np.ndim(out) == 0 else out

    def is_allowed(self, direction: ArrayLike) -> NDArray[np.bool_] | bool:
        """``True`` where the direction violates no cone."""
        m = self.margins(direction)
        if m.shape[-1] == 0:
            return True if m.ndim == 1 else np.ones(m.shape[:-1], dtype=bool)
        out = np.all(m >= 0.0, axis=-1)
        return bool(out) if np.ndim(out) == 0 else out

    def violations(self, direction: ArrayLike) -> tuple[str, ...]:
        """Names of cones a single direction violates, deepest first."""
        n = unit_vector(direction).reshape(3)
        m = self.margins(n)
        idx = [int(i) for i in np.argsort(m) if m[i] < 0.0]
        return tuple(self.cones[i].name for i in idx)

    def rotated(self, rotation: ArrayLike) -> KeepOutSet:
        """The whole set with every axis rotated by a ``(3, 3)`` matrix."""
        return KeepOutSet(tuple(c.rotated(rotation) for c in self.cones))

    def covers_sphere(self) -> bool:
        """``True`` if a single cone already excludes every direction.

        Only the trivial ``half_angle >= pi`` case is decided here; a union of
        smaller cones covering the sphere is not detected, and the planner
        reports that situation as "no feasible path found" instead.
        """
        return any(c.half_angle >= math.pi - _ANG_EPS for c in self.cones)

    # -- arc queries -----------------------------------------------------

    def min_margin_on_arc(self, n0: ArrayLike, eigenaxis: ArrayLike, sweep: float) -> float:
        """Smallest clearance [rad] over all cones and the whole arc.

        ``+inf`` for an empty set. Closed form per cone; see
        :meth:`KeepOutCone.min_margin_on_arc`.
        """
        if not self.cones:
            return float("inf")
        return min(c.min_margin_on_arc(n0, eigenaxis, sweep) for c in self.cones)

    def arc_violations(
        self,
        n0: ArrayLike,
        eigenaxis: ArrayLike,
        sweep: float,
        boresight_name: str = "boresight",
    ) -> tuple[ArcViolation, ...]:
        """Every violating stretch of the arc, deepest first.

        Parameters
        ----------
        n0, eigenaxis : array_like
            Shape ``(3,)``, inertial: boresight at ``psi = 0`` and the swept
            axis.
        sweep : float
            Total swept angle [rad].
        boresight_name : str
            Recorded in each :class:`ArcViolation` so a multi-instrument
            report says which instrument was affected.
        """
        out: list[ArcViolation] = []
        for cone in self.cones:
            a, b, c = cone.arc_coefficients(n0, eigenaxis)
            for lo, hi in cone.violation_intervals(n0, eigenaxis, sweep):
                fmax = max(
                    a * math.cos(lo) + b * math.sin(lo) + c,
                    a * math.cos(hi) + b * math.sin(hi) + c,
                )
                r = math.hypot(a, b)
                if r > _ANG_EPS:
                    phi = math.atan2(b, a)
                    for k in range(-2, 3):
                        psi = phi + 2.0 * math.pi * k
                        if lo <= psi <= hi:
                            fmax = max(fmax, r + c)
                            break
                fmax = min(1.0, max(-1.0, fmax))
                depth = cone.half_angle - math.acos(fmax)
                out.append(ArcViolation(cone.name, boresight_name, lo, hi, depth))
        out.sort(key=lambda v: -v.depth)
        return tuple(out)

    def sample_min_margin_on_arc(
        self, n0: ArrayLike, eigenaxis: ArrayLike, sweep: float, n_samples: int = 2001
    ) -> float:
        """Sampled minimum clearance [rad] -- the naive method, kept for testing.

        This is what a planner that samples the path computes. It is used only
        as an independent cross-check of :meth:`min_margin_on_arc` in the test
        suite and validation scripts; nothing in the planner calls it.
        """
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        if not self.cones:
            return float("inf")
        psi = np.linspace(0.0, float(sweep), int(n_samples))
        pts = rotate_about_axis(n0, eigenaxis, psi)
        return float(np.min(self.margins(pts)))
