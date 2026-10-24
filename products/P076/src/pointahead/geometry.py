"""Vector geometry for a single inertial frame, in SI units throughout.

Conventions used by every module in this package:

* A direction or position is a length-3 ``numpy`` array of ``float64``.
* Positions are metres, velocities metres per second, times seconds, angles
  radians. Nothing in this package uses kilometres, degrees or days; the YAML
  case loader is the only place a non-SI input is accepted and it converts at
  the boundary.
* One frame at a time. Every function here is frame-agnostic, and the caller is
  responsible for not mixing frames. The point-ahead decomposition in
  :mod:`pointahead.angle` is frame-dependent term by term (see its module
  docstring) and that is stated where it matters rather than hidden here.

The speed of light is the exact SI defining constant, 299792458 m/s
(BIPM, SI Brochure, 9th edition, 2019, definition of the metre). It is exact by
definition, so it carries no uncertainty line in the error budget.
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 299792458.0
"""Speed of light in vacuum, m/s. Exact by SI definition."""


def as_vector(value: object, name: str = "vector") -> np.ndarray:
    """Coerce ``value`` to a finite length-3 float64 array.

    Parameters
    ----------
    value
        Anything ``numpy`` can read as a length-3 sequence.
    name
        Used in the error message so a caller knows which argument failed.

    Raises
    ------
    TypeError
        If ``value`` cannot be read as a float array.
    ValueError
        If the result is not length 3, or contains a NaN or an infinity.
    """
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:  # pragma: no cover - numpy message varies
        raise TypeError(f"{name} must be a length-3 sequence of floats, got {value!r}") from exc
    if array.size != 3:
        raise ValueError(f"{name} must have exactly 3 components, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array.tolist()}")
    return array


def norm(vector: object) -> float:
    """Euclidean norm of a length-3 vector, in the units of the input."""
    return float(np.linalg.norm(as_vector(vector)))


def unit(vector: object, name: str = "vector") -> np.ndarray:
    """Unit vector along ``vector``.

    Raises
    ------
    ValueError
        If the norm is zero or underflows to zero, because a direction is then
        undefined. Returning an arbitrary axis here would silently corrupt a
        point-ahead angle, so it raises instead.
    """
    array = as_vector(vector, name)
    magnitude = float(np.linalg.norm(array))
    if magnitude == 0.0:
        raise ValueError(f"{name} has zero norm, so it has no direction")
    result = array / magnitude
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} norm underflowed, cannot normalise {array.tolist()}")
    return result


def transverse(vector: object, direction: object) -> np.ndarray:
    """Component of ``vector`` perpendicular to ``direction``.

    ``direction`` need not be normalised; it is normalised internally. The
    returned vector has the units of ``vector``.

    This is the projection ``v - (v . n) n`` with ``n`` the unit of
    ``direction``. It is the operation that turns a relative velocity into the
    transverse relative velocity that drives the point-ahead angle.
    """
    v = as_vector(vector, "vector")
    n = unit(direction, "direction")
    return v - float(np.dot(v, n)) * n


def radial(vector: object, direction: object) -> float:
    """Signed component of ``vector`` along ``direction``, units of ``vector``."""
    v = as_vector(vector, "vector")
    n = unit(direction, "direction")
    return float(np.dot(v, n))


def angle_between(first: object, second: object) -> float:
    """Angle between two vectors in radians, in ``[0, pi]``.

    Uses ``atan2(|a x b|, a . b)``. The ``arccos`` of a normalised dot product
    loses all precision for small angles -- at 1e-8 rad the dot product is
    ``1 - 5e-17``, which rounds to 1.0 in float64 and returns exactly zero.
    Every angle this package reports is of order 1e-5 rad or smaller, so the
    cross-product form is not a refinement here, it is the only usable one.
    """
    a = as_vector(first, "first")
    b = as_vector(second, "second")
    cross = float(np.linalg.norm(np.cross(a, b)))
    dot = float(np.dot(a, b))
    if cross == 0.0 and dot == 0.0:
        raise ValueError("angle_between needs two non-zero vectors")
    return float(np.arctan2(cross, dot))


def rotation_matrix(axis: object, angle_rad: float) -> np.ndarray:
    """Right-handed rotation matrix about ``axis`` by ``angle_rad``.

    Rodrigues' rotation formula. Used by the tests and validation scripts to
    rotate an entire crosslink configuration, which must leave every angle this
    package computes unchanged.
    """
    k = unit(axis, "axis")
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError(f"angle_rad must be finite, got {angle_rad!r}")
    kx = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def orthonormal_basis(direction: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed basis ``(n, e1, e2)`` with ``n`` along ``direction``.

    ``e1`` and ``e2`` span the transverse plane. The choice of ``e1`` is
    arbitrary but deterministic: it is built from whichever Cartesian axis is
    least aligned with ``n``, so the basis is stable and never degenerate.
    Point-ahead *magnitudes* do not depend on this choice; the two-component
    transverse offset reported alongside them does, and that is said in
    :func:`pointahead.angle.reference_point_ahead`.
    """
    n = unit(direction, "direction")
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(n)))] = 1.0
    e1 = unit(seed - float(np.dot(seed, n)) * n, "basis seed")
    e2 = np.cross(n, e1)
    return n, e1, e2
