"""Allowed-attitude regions: which boresight directions survive a keep-out set.

Two questions are answered here. *Which* directions are allowed --
:func:`allowed_mask` and :func:`allowed_directions` -- and *how much* of the sky
is allowed, as a solid angle in steradians.

The size question has an exact answer in azimuth. At a fixed colatitude
``theta`` each exclusion cone cuts a single arc out of the ring of constant
``theta``, and the arc's half-width follows from the spherical law of cosines.
Taking the union of those arcs and integrating the leftover azimuth measure
against ``d(cos theta)`` reduces a two-dimensional problem to a
one-dimensional quadrature that is exact in one of the two coordinates. The
quadrature nodes are placed between the colatitudes where the integrand's
derivative jumps, which are the caps' extreme latitudes and the latitudes of
the cap-boundary crossings.

Angles are radians, solid angles steradians.

References
----------
I. Todhunter, *Spherical Trigonometry*, 5th ed., Macmillan (1886), Art. 37 --
    the law of cosines that gives the arc half-width.
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 5.2 -- attitude cone geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cones import KeepOutSet
from .geometry import fibonacci_sphere, unit

__all__ = [
    "SolidAngleEstimate",
    "allowed_mask",
    "allowed_directions",
    "allowed_solid_angle",
    "allowed_fraction",
    "allowed_solid_angle_monte_carlo",
]

_FULL_SPHERE_SR = 4.0 * np.pi


@dataclass(frozen=True)
class SolidAngleEstimate:
    """A solid angle with an uncertainty and the sample count behind it.

    Attributes
    ----------
    solid_angle : float
        Estimate [sr].
    standard_error : float
        One-sigma uncertainty [sr]. Zero for deterministic quadrature.
    n_samples : int
        Number of samples or quadrature nodes used.
    """

    solid_angle: float
    standard_error: float
    n_samples: int

    @property
    def fraction(self) -> float:
        """Estimate as a fraction of the full sphere (dimensionless)."""
        return self.solid_angle / _FULL_SPHERE_SR

    @property
    def fraction_standard_error(self) -> float:
        """One-sigma uncertainty on :attr:`fraction` (dimensionless)."""
        return self.standard_error / _FULL_SPHERE_SR


def allowed_mask(keepout: KeepOutSet, directions: ArrayLike) -> NDArray[np.bool_]:
    """Boolean mask of directions that violate no cone.

    Parameters
    ----------
    keepout : KeepOutSet
    directions : array_like
        Shape ``(..., 3)``; normalised internally.

    Returns
    -------
    ndarray of bool
        Shape ``(...)``.
    """
    out = keepout.is_allowed(directions)
    return np.atleast_1d(np.asarray(out, dtype=bool))


def allowed_directions(keepout: KeepOutSet, n_points: int = 4000) -> NDArray[np.float64]:
    """Allowed directions sampled from a Fibonacci lattice on the sphere.

    Parameters
    ----------
    keepout : KeepOutSet
    n_points : int
        Lattice size before masking, ``>= 1``.

    Returns
    -------
    ndarray
        Shape ``(m, 3)`` with ``m <= n_points``; empty if nothing is allowed.
    """
    pts = fibonacci_sphere(n_points)
    return pts[allowed_mask(keepout, pts)]


def _cap_parameters(keepout: KeepOutSet) -> tuple[NDArray, NDArray, NDArray]:
    if not keepout.cones:
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    axes = np.array([c.axis for c in keepout.cones], dtype=float)
    radii = np.array([c.half_angle for c in keepout.cones], dtype=float)
    return axes, radii, np.arccos(np.clip(axes[:, 2], -1.0, 1.0))


def _blocked_azimuth_measure(
    u: float, axes: NDArray, radii: NDArray, colats: NDArray
) -> float:
    """Azimuth measure blocked by the caps on the ring ``cos(theta) = u`` [rad]."""
    sin_t = np.sqrt(max(0.0, 1.0 - u * u))
    intervals: list[tuple[float, float]] = []
    for axis, r, tc in zip(axes, radii, colats, strict=True):
        cos_tc = axis[2]
        sin_tc = np.sqrt(max(0.0, 1.0 - cos_tc * cos_tc))
        denom = sin_t * sin_tc
        num = np.cos(r) - u * cos_tc
        if denom < 1e-14:
            # Ring or cap axis at a pole: the whole ring is in or out together.
            if u * cos_tc >= np.cos(r) - 1e-14:
                return 2.0 * np.pi
            continue
        x = num / denom
        if x <= -1.0:
            return 2.0 * np.pi
        if x >= 1.0:
            continue
        half = float(np.arccos(x))
        phi_c = float(np.arctan2(axis[1], axis[0]))
        intervals.append((phi_c - half, phi_c + half))
        del tc
    if not intervals:
        return 0.0
    # Union of arcs on a circle: unwrap to [0, 2pi), split wrapping arcs, sweep.
    spans: list[tuple[float, float]] = []
    two_pi = 2.0 * np.pi
    for lo, hi in intervals:
        lo_m = float(np.mod(lo, two_pi))
        hi_m = lo_m + (hi - lo)
        if hi_m > two_pi:
            spans.append((lo_m, two_pi))
            spans.append((0.0, hi_m - two_pi))
        else:
            spans.append((lo_m, hi_m))
    spans.sort()
    total = 0.0
    cur_lo, cur_hi = spans[0]
    for lo, hi in spans[1:]:
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    total += cur_hi - cur_lo
    return min(total, two_pi)


def _breakpoints(axes: NDArray, radii: NDArray, colats: NDArray) -> NDArray[np.float64]:
    """Values of ``u = cos(theta)`` where the azimuth measure has a kink."""
    pts = [-1.0, 1.0]
    for tc, r in zip(colats, radii, strict=True):
        # The arc half-width arccos(x) has an infinite derivative where the ring
        # first touches the cap (x = +1, theta = tc -+ r) and where the ring
        # becomes wholly enclosed by it (x = -1, theta + tc = r near the north
        # pole or theta + tc = 2 pi - r near the south pole).
        for theta in (tc - r, tc + r, r - tc, 2.0 * np.pi - r - tc):
            if 0.0 <= theta <= np.pi:
                pts.append(np.cos(theta))
    n = len(axes)
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = axes[i], axes[j]
            c = float(np.dot(a1, a2))
            s2 = 1.0 - c * c
            if s2 < 1e-14:
                continue
            c1, c2 = np.cos(radii[i]), np.cos(radii[j])
            x = (c1 - c * c2) / s2
            y = (c2 - c * c1) / s2
            cross = np.cross(a1, a2)
            norm2 = float(np.dot(cross, cross))
            z2 = (1.0 - (x * x + y * y + 2.0 * x * y * c)) / norm2
            if z2 < 0.0:
                continue
            z = np.sqrt(z2)
            for sgn in (+1.0, -1.0):
                p = x * a1 + y * a2 + sgn * z * cross
                pts.append(float(np.clip(p[2], -1.0, 1.0)))
    u = np.unique(np.clip(np.asarray(pts, dtype=float), -1.0, 1.0))
    return u


def allowed_solid_angle(keepout: KeepOutSet, nodes_per_band: int = 96) -> SolidAngleEstimate:
    """Solid angle of the allowed sky [sr], by band quadrature.

    Exact in azimuth (the blocked arc on each ring is a closed form), and
    Gauss-Legendre in ``u = cos(theta)`` on each interval between kinks in the
    integrand. Kinks sit at the caps' extreme colatitudes and at the colatitudes
    where two cap boundaries cross, so the integrand is smooth on the interior
    of every subinterval.

    At a subinterval endpoint the blocked measure still behaves like
    ``sqrt(|u - u_edge|)``, because the arc half-width ``arccos(x)`` has an
    infinite derivative where ``x`` reaches ``+-1``. Plain Gauss-Legendre then
    converges only algebraically. The substitution ``u = mid + half sin(s)``
    with ``s`` on ``[-pi/2, pi/2]`` supplies a ``cos(s)`` Jacobian that cancels
    the square root and clusters nodes at the endpoints, restoring fast
    convergence; measured in ``validation/validate_cone_geometry.py``.

    Parameters
    ----------
    keepout : KeepOutSet
    nodes_per_band : int
        Gauss-Legendre nodes per subinterval, ``>= 2``. Cost is
        ``nodes_per_band * n_bands * n_cones``. The default of 96 gives a worst
        error of 8.7e-14 sr over 300 random two-cap configurations against the
        closed form; 24 nodes gives 1.1e-06 sr and 48 gives 1.2e-08 sr
        (``validation/validate_cone_geometry.py``).

    Returns
    -------
    SolidAngleEstimate
        ``standard_error`` is ``0.0`` -- this estimator is deterministic. Use
        :func:`allowed_solid_angle_monte_carlo` if an independent estimate with
        a statistical error bar is wanted.

    Notes
    -----
    Validity: any number of cones, any radii, including full overlap. Verified
    against the two-cap closed form in
    ``validation/validate_cone_geometry.py``.
    """
    if nodes_per_band < 2:
        raise ValueError(f"nodes_per_band must be >= 2, got {nodes_per_band}")
    axes, radii, colats = _cap_parameters(keepout)
    if len(axes) == 0:
        return SolidAngleEstimate(_FULL_SPHERE_SR, 0.0, 0)
    edges = _breakpoints(axes, radii, colats)
    x_ref, w_ref = np.polynomial.legendre.leggauss(nodes_per_band)
    s_ref = 0.5 * np.pi * x_ref
    ws_ref = 0.5 * np.pi * w_ref
    total = 0.0
    n_used = 0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi - lo < 1e-15:
            continue
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        for si, wi in zip(s_ref, ws_ref, strict=True):
            u = mid + half * np.sin(si)
            jac = half * np.cos(si)
            blocked = _blocked_azimuth_measure(float(u), axes, radii, colats)
            total += wi * jac * (2.0 * np.pi - blocked)
            n_used += 1
    return SolidAngleEstimate(float(total), 0.0, n_used)


def allowed_fraction(keepout: KeepOutSet, nodes_per_band: int = 96) -> float:
    """Allowed sky as a fraction of the full sphere (dimensionless, 0 to 1)."""
    return allowed_solid_angle(keepout, nodes_per_band=nodes_per_band).fraction


def allowed_solid_angle_monte_carlo(
    keepout: KeepOutSet, n_samples: int = 200_000, seed: int | None = None
) -> SolidAngleEstimate:
    """Independent Monte Carlo estimate of the allowed solid angle [sr].

    Directions are drawn uniformly on the sphere by normalising isotropic
    Gaussian vectors (Muller, *Comm. ACM* 2(4), 19-20, 1959). The hit fraction
    ``p`` is binomial, so the one-sigma error on the solid angle is
    ``4 pi sqrt(p (1 - p) / n)``.

    Parameters
    ----------
    keepout : KeepOutSet
    n_samples : int
        Sample count, ``>= 1``.
    seed : int, optional
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    SolidAngleEstimate
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    pts = unit(rng.normal(size=(n_samples, 3)))
    hits = int(np.count_nonzero(allowed_mask(keepout, pts)))
    p = hits / n_samples
    err = _FULL_SPHERE_SR * np.sqrt(max(p * (1.0 - p), 0.0) / n_samples)
    return SolidAngleEstimate(_FULL_SPHERE_SR * p, float(err), n_samples)
