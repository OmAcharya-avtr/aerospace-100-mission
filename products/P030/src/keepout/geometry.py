"""Spherical geometry primitives for celestial keep-out cones.

Everything in this module is dimensionless direction geometry on the unit
two-sphere. Angles are radians unless a name says ``_deg``. Directions are
Cartesian unit vectors in whatever right-handed frame the caller is using;
the module never assumes a particular frame.

References
----------
I. Todhunter, *Spherical Trigonometry*, 5th ed., Macmillan (1886) --
    the spherical law of cosines used throughout, Art. 37.
D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
    Microcosm Press (2013) -- vector/angle conventions, Appendix C.
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*,
    Reidel (1978) -- attitude geometry and cone/angular-separation usage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "unit",
    "angular_separation",
    "rotation_matrix",
    "random_rotations",
    "spherical_to_unit",
    "unit_to_spherical",
    "cap_solid_angle",
    "cap_intersection_solid_angle",
    "cap_union_solid_angle",
    "fibonacci_sphere",
]

_EPS = 1e-15
_NESTED_TOL = 1e-12


def unit(v: ArrayLike) -> NDArray[np.float64]:
    """Normalise a vector or a stack of vectors to unit length.

    Parameters
    ----------
    v : array_like
        Shape ``(3,)`` or ``(..., 3)``. Any units; the units cancel.

    Returns
    -------
    ndarray
        Same shape, unit norm along the last axis (dimensionless).

    Raises
    ------
    ValueError
        If the last axis is not length 3, if any entry is non-finite, or if
        any vector has norm below 1e-13 (direction undefined).
    """
    a = np.asarray(v, dtype=float)
    if a.ndim == 0 or a.shape[-1] != 3:
        raise ValueError(f"expected shape (...,3), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("vector contains non-finite entries")
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-13):
        raise ValueError("cannot normalise a vector of norm < 1e-13; direction is undefined")
    return a / n


def angular_separation(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64] | float:
    """Angle between two directions [rad], in ``[0, pi]``.

    Uses ``atan2(|a x b|, a . b)`` rather than ``arccos(a . b)``. The dot-product
    form loses roughly half the significant digits for nearly parallel vectors
    (its derivative diverges as the angle goes to zero); the cross-product form
    is accurate over the whole range. See Vallado (2013), Appendix C, on the
    conditioning of the arccosine form.

    Parameters
    ----------
    a, b : array_like
        Shape ``(3,)`` or ``(..., 3)``; broadcast against each other. Need not
        be normalised.

    Returns
    -------
    float or ndarray
        Angle in radians, in ``[0, pi]``.
    """
    ua = unit(a)
    ub = unit(b)
    cross = np.linalg.norm(np.cross(ua, ub), axis=-1)
    dot = np.sum(ua * ub, axis=-1)
    out = np.arctan2(cross, dot)
    return float(out) if np.ndim(out) == 0 else out


def rotation_matrix(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Right-handed rotation matrix about ``axis`` by ``angle`` [rad].

    Rodrigues' rotation formula, ``R = I + sin(t) K + (1 - cos(t)) K^2`` with
    ``K`` the cross-product matrix of the unit axis. Active convention: ``R @ v``
    rotates the vector ``v`` within a fixed frame. See Wertz (1978), Sec. 12.1.

    Parameters
    ----------
    axis : array_like
        Rotation axis, shape ``(3,)``; normalised internally.
    angle : float
        Rotation angle [rad]. Any real value.

    Returns
    -------
    ndarray
        Shape ``(3, 3)``, orthogonal with determinant ``+1``.
    """
    k = unit(axis)
    kx, ky, kz = k
    kmat = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * kmat + (1.0 - np.cos(angle)) * (kmat @ kmat)


def random_rotations(n: int, seed: int | None = None) -> NDArray[np.float64]:
    """``n`` rotation matrices drawn uniformly from SO(3) (Haar measure).

    Built from unit quaternions sampled uniformly on the 3-sphere, which is the
    Haar measure on SO(3) (Shoemake, *Graphics Gems III*, 1992, "Uniform Random
    Rotations").

    Parameters
    ----------
    n : int
        Number of matrices, ``n >= 1``.
    seed : int, optional
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    ndarray
        Shape ``(n, 3, 3)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((n, 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def spherical_to_unit(right_ascension: ArrayLike, declination: ArrayLike) -> NDArray[np.float64]:
    """Convert (right ascension, declination) [rad] to a unit vector.

    ``x`` towards ``(0, 0)``, ``z`` towards the ``+90 deg`` pole; the same
    convention as an equatorial frame with ``+x`` at the vernal equinox
    (Vallado 2013, Sec. 3.4).

    Returns
    -------
    ndarray
        Shape ``(..., 3)``.
    """
    ra = np.asarray(right_ascension, dtype=float)
    dec = np.asarray(declination, dtype=float)
    if np.any(np.abs(dec) > np.pi / 2 + 1e-12):
        raise ValueError("declination must lie in [-pi/2, pi/2] rad")
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


def unit_to_spherical(v: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of :func:`spherical_to_unit`.

    Returns
    -------
    (ra, dec) : tuple of ndarray
        Right ascension in ``[0, 2 pi)`` and declination in ``[-pi/2, pi/2]``,
        both radians.
    """
    u = unit(v)
    ra = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
    dec = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    return ra, dec


def cap_solid_angle(half_angle: ArrayLike) -> NDArray[np.float64] | float:
    """Solid angle of a spherical cap of angular radius ``half_angle`` [sr].

    ``Omega = 2 pi (1 - cos(alpha))``, the elementary integral
    ``int_0^alpha 2 pi sin(theta) d(theta)``. Exact for ``0 <= alpha <= pi``;
    ``alpha = pi/2`` gives ``2 pi`` (hemisphere) and ``alpha = pi`` gives
    ``4 pi`` (whole sphere).

    Parameters
    ----------
    half_angle : array_like
        Angular radius [rad], ``0 <= alpha <= pi``.

    Returns
    -------
    float or ndarray
        Solid angle [sr].
    """
    a = np.asarray(half_angle, dtype=float)
    if np.any(a < 0.0) or np.any(a > np.pi):
        raise ValueError("half_angle must lie in [0, pi] rad")
    out = 2.0 * np.pi * (1.0 - np.cos(a))
    return float(out) if np.ndim(out) == 0 else out


def cap_intersection_solid_angle(r1: float, r2: float, separation: float) -> float:
    """Closed-form solid angle common to two spherical caps [sr].

    Caps of angular radii ``r1``, ``r2`` whose axes are separated by
    ``separation = d``. For the properly-overlapping case
    ``|r1 - r2| < d < r1 + r2`` the lens area follows from the Gauss-Bonnet
    theorem on the unit sphere (``K = 1``)::

        A + integral(k_g ds) + sum(exterior angles) = 2 pi

    A small circle of angular radius ``r`` has geodesic curvature ``cot(r)`` and
    the bounding arc subtending half-azimuth ``alpha`` at its own centre has
    length ``2 alpha sin(r)``, so each arc contributes ``2 alpha_i cos(r_i)``.
    The two vertices each contribute an exterior angle ``pi - (pi - gamma)``,
    giving::

        A = 2 (pi - gamma) - 2 alpha_1 cos(r1) - 2 alpha_2 cos(r2)

        cos(alpha_1) = (cos r2 - cos r1 cos d) / (sin r1 sin d)
        cos(alpha_2) = (cos r1 - cos r2 cos d) / (sin r2 sin d)
        cos(gamma)   = (cos d  - cos r1 cos r2) / (sin r1 sin r2)

    The three auxiliary angles are the spherical law of cosines (Todhunter 1886,
    Art. 37) applied to the triangle (axis 1, axis 2, boundary crossing point).

    Three degenerate cases are handled before that formula is reached, because
    the spherical triangle it relies on does not exist in them:

    - ``d >= r1 + r2``: the caps are disjoint, intersection ``0``.
    - ``d <= |r1 - r2|``: one cap is wholly inside the other, intersection is
      the smaller cap. Compared with a ``1e-12`` rad tolerance, because the
      lens formula divides by ``sin(d)`` and loses all its significance as the
      axes coincide.
    - ``r1 + r2 + d >= 2 pi``: the *complement* caps, of radii ``pi - r1`` and
      ``pi - r2`` about the reversed axes, are disjoint, so the two caps cover
      the whole sphere and the intersection is ``A1 + A2 - 4 pi``. This is the
      fourth condition for a spherical triangle to exist -- the perimeter must
      not exceed ``2 pi`` -- and omitting it makes the lens formula return a
      value below the true intersection for two very large caps.

    Parameters
    ----------
    r1, r2 : float
        Cap angular radii [rad], each in ``[0, pi]``.
    separation : float
        Angle between the two cap axes [rad], in ``[0, pi]``.

    Returns
    -------
    float
        Intersection solid angle [sr], in ``[0, 4 pi]``.

    Notes
    -----
    Validity: exact for all ``r1, r2 in [0, pi]`` and ``d in [0, pi]`` on the
    unit sphere. Verified against band quadrature and Monte Carlo integration in
    ``validation/validate_cone_geometry.py``.
    """
    for name, val in (("r1", r1), ("r2", r2)):
        if not 0.0 <= val <= np.pi:
            raise ValueError(f"{name} must lie in [0, pi] rad, got {val}")
    if not 0.0 <= separation <= np.pi:
        raise ValueError(f"separation must lie in [0, pi] rad, got {separation}")
    d = float(separation)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2) + _NESTED_TOL:
        return float(cap_solid_angle(min(r1, r2)))
    if r1 + r2 + d >= 2.0 * np.pi:
        return float(cap_solid_angle(r1) + cap_solid_angle(r2) - 4.0 * np.pi)
    c1, c2 = np.cos(r1), np.cos(r2)
    a1 = _triangle_angle(r2, r1, d)
    a2 = _triangle_angle(r1, r2, d)
    gam = _triangle_angle(d, r1, r2)
    return float(2.0 * (np.pi - gam) - 2.0 * a1 * c1 - 2.0 * a2 * c2)


def _triangle_angle(a: float, b: float, c: float) -> float:
    """Angle opposite side ``a`` in a spherical triangle with sides ``a, b, c``.

    Half-angle form, Todhunter (1886), Art. 45, with ``s = (a + b + c) / 2``::

        tan^2(A / 2) = sin(s - b) sin(s - c) / (sin s sin(s - a))

    Every argument is the sine of a positive quantity smaller than ``pi``
    whenever the triangle exists, so nothing cancels. The direct law-of-cosines
    form ``cos A = (cos a - cos b cos c) / (sin b sin c)`` loses all its
    significant digits for triangles whose sides are near the square root of the
    machine epsilon, roughly 1e-8 rad; this form does not.
    """
    s = 0.5 * (a + b + c)
    num = np.sin(max(s - b, 0.0)) * np.sin(max(s - c, 0.0))
    den = np.sin(s) * np.sin(max(s - a, 0.0))
    if den <= _EPS:  # pragma: no cover - excluded by the degenerate branches
        return 0.0 if num <= _EPS else np.pi
    return float(2.0 * np.arctan(np.sqrt(num / den)))


def cap_union_solid_angle(r1: float, r2: float, separation: float) -> float:
    """Solid angle covered by either of two spherical caps [sr].

    ``A1 + A2 - A_intersection``; exact for two caps, from
    :func:`cap_solid_angle` and :func:`cap_intersection_solid_angle`.
    """
    return float(
        cap_solid_angle(r1)
        + cap_solid_angle(r2)
        - cap_intersection_solid_angle(r1, r2, separation)
    )


def fibonacci_sphere(n: int) -> NDArray[np.float64]:
    """``n`` approximately equal-area points on the unit sphere.

    Spherical Fibonacci lattice: ``z`` equispaced in ``[-1, 1]`` (equal-area in
    the polar direction) with azimuth advanced by the golden angle
    ``pi (3 - sqrt(5))``. See Gonzalez, "Measurement of Areas on a Sphere Using
    Fibonacci and Latitude-Longitude Lattices", *Mathematical Geosciences* 42,
    49-64 (2010). Discrepancy falls roughly as ``n^-1``, better than a
    latitude-longitude grid and better than random sampling's ``n^-1/2``.

    Parameters
    ----------
    n : int
        Number of points, ``n >= 1``.

    Returns
    -------
    ndarray
        Shape ``(n, 3)``, unit vectors.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    i = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)
