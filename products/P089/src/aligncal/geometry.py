"""Small-rotation algebra for relative sensor alignment.

Everything in this module is SO(3) bookkeeping. It is separated out because
every other module depends on exactly one convention and a sign error here
would be invisible everywhere else.

Conventions, fixed once
-----------------------
A misalignment is carried as a rotation **vector** ``theta`` in radians, in the
frame of the reference sensor, and the corresponding rotation matrix is

    (1)   R(theta) = exp([theta]x)

with ``[a]x`` the skew-symmetric cross-product matrix of ``a``, so that
``[a]x b = a x b``. The closed form of (1) is Rodrigues' formula

    (2)   R(theta) = I + (sin p / p) [theta]x + ((1 - cos p) / p^2) [theta]x^2,
          p = |theta|

Source: O. Rodrigues (1840), as given in standard form by Markley, F. L. and
Crassidis, J. L. (2014), *Fundamentals of Spacecraft Attitude Determination and
Control*, Springer, equation (2.113). Units: ``theta`` in radians, ``R``
dimensionless. Validity: all ``theta``; the series forms used here are switched
to their Taylor limits below ``p = 1e-8`` rad where the closed forms lose
significance to cancellation.

The identity this product actually rests on is

    (3)   [a]x^T [a]x = -[a]x [a]x = |a|^2 I - a a^T

which is why the observability Gramian of a rate-pair alignment problem has the
rotation axis in its null space. It is proved by direct expansion and is
exercised as a property test.

Validity range of the small-angle linearisation used by the estimators. Two
different relative errors are involved and conflating them is a trap this
module's own test suite fell into once:

* the rotation **matrix** ``R(theta) ~ I + [theta]x`` has absolute error
  ``|theta|^2 / 2`` against a matrix of order 1, so its relative error is
  ``|theta|^2 / 2``: about 5e-5 at ``|theta| = 10 mrad``;
* the **residual** ``r = [w]x theta`` is itself of order ``|theta| |w|``, so
  the same absolute error is a relative error of ``|theta| / 2``: about 5e-3 at
  ``|theta| = 10 mrad``, a hundred times larger.

The second is the one that matters for an estimate. The batch estimator and the
MEKF both iterate, re-linearising about the current estimate, and therefore do
not inherit it; a single linear pass and the learned estimator do. The figures
are quoted in README.md under Limitations.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SMALL_ANGLE_RAD",
    "angle_between",
    "is_rotation",
    "rotation_angle",
    "rotation_from_vector",
    "rotation_to_vector",
    "skew",
    "unskew",
]

#: Below this rotation magnitude the Taylor limits of (2) are used instead of
#: the closed forms, which lose significance to cancellation. Radians.
SMALL_ANGLE_RAD = 1.0e-8


def skew(vector: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix ``[a]x`` of ``a``.

    Parameters
    ----------
    vector:
        Shape ``(3,)`` or ``(..., 3)``. Any unit; the output carries the same.

    Returns
    -------
    ndarray
        Shape ``(..., 3, 3)``, satisfying ``skew(a) @ b == np.cross(a, b)``.
    """
    array = np.asarray(vector, dtype=float)
    if array.shape[-1] != 3:
        raise ValueError(f"skew() needs a last axis of length 3, got shape {array.shape}")
    zero = np.zeros(array.shape[:-1])
    a0, a1, a2 = array[..., 0], array[..., 1], array[..., 2]
    rows = [
        np.stack([zero, -a2, a1], axis=-1),
        np.stack([a2, zero, -a0], axis=-1),
        np.stack([-a1, a0, zero], axis=-1),
    ]
    return np.stack(rows, axis=-2)


def unskew(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew` on the skew-symmetric part of ``matrix``.

    The skew-symmetric part is taken rather than assumed, so a matrix carrying
    rounding-level symmetry is handled without raising. Units follow the input.
    """
    array = np.asarray(matrix, dtype=float)
    if array.shape[-2:] != (3, 3):
        raise ValueError(f"unskew() needs a trailing (3, 3), got shape {array.shape}")
    anti = 0.5 * (array - np.swapaxes(array, -1, -2))
    return np.stack([anti[..., 2, 1], anti[..., 0, 2], anti[..., 1, 0]], axis=-1)


def rotation_from_vector(theta: np.ndarray) -> np.ndarray:
    """Rotation matrix ``R = exp([theta]x)`` by equation (2). ``theta`` in radians.

    Returns a dimensionless ``(3, 3)`` matrix with ``det = +1``.
    """
    vector = np.asarray(theta, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"rotation_from_vector() needs 3 components, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("rotation_from_vector() needs a finite rotation vector")
    magnitude = float(np.linalg.norm(vector))
    cross = skew(vector)
    if magnitude < SMALL_ANGLE_RAD:
        # Taylor limits of (2): sin p / p -> 1 - p^2/6, (1 - cos p)/p^2 -> 1/2.
        return np.eye(3) + cross + 0.5 * (cross @ cross)
    first = np.sin(magnitude) / magnitude
    second = (1.0 - np.cos(magnitude)) / (magnitude * magnitude)
    return np.eye(3) + first * cross + second * (cross @ cross)


def rotation_angle(rotation: np.ndarray) -> float:
    """Rotation angle of ``rotation`` in radians, in ``[0, pi]``.

    ``trace(R) = 1 + 2 cos p``; the argument is clipped because a matrix that is
    orthogonal only to rounding can push it marginally outside ``[-1, 1]``.
    """
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_angle() needs a (3, 3) matrix, got {matrix.shape}")
    cosine = 0.5 * (float(np.trace(matrix)) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation_to_vector(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector ``theta`` with ``exp([theta]x) == rotation``, radians.

    The principal logarithm is returned, so ``|theta| <= pi``. Near ``p = pi``
    the skew part degenerates and the axis is recovered from the symmetric part
    ``R + R^T``, which is the standard branch handling; this product never
    operates near that branch, but a silent wrong answer there would be worse
    than a slower correct one.
    """
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_to_vector() needs a (3, 3) matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation_to_vector() needs a finite matrix")
    angle = rotation_angle(matrix)
    if angle < SMALL_ANGLE_RAD:
        return unskew(matrix)
    if np.pi - angle < 1.0e-6:
        symmetric = 0.5 * (matrix + matrix.T) - np.cos(angle) * np.eye(3)
        scale = 1.0 - np.cos(angle)
        column = int(np.argmax(np.diag(symmetric)))
        axis = symmetric[:, column] / np.sqrt(max(symmetric[column, column] * scale, 1e-300))
        axis = axis / float(np.linalg.norm(axis))
        # Sign is ambiguous at exactly pi; fix it from the skew part when it is
        # resolvable and leave the positive branch otherwise.
        skew_part = unskew(matrix)
        if float(np.dot(skew_part, axis)) < 0.0:
            axis = -axis
        return angle * axis
    return unskew(matrix) * (angle / np.sin(angle))


def is_rotation(matrix: np.ndarray, tolerance: float = 1.0e-9) -> bool:
    """True when ``matrix`` is orthogonal with ``det = +1`` to ``tolerance``.

    The tolerance is absolute, on the largest element of ``R^T R - I`` and on
    ``det - 1``. Both are dimensionless.
    """
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        return False
    orthogonality = float(np.max(np.abs(array.T @ array - np.eye(3))))
    determinant = float(np.linalg.det(array))
    return orthogonality <= tolerance and abs(determinant - 1.0) <= tolerance


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """Angle between two vectors in radians, in ``[0, pi]``.

    Raises ``ValueError`` on a zero-length vector, for which the angle is not
    defined; returning zero there would be a silent wrong answer.
    """
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    if a.size != 3 or b.size != 3:
        raise ValueError("angle_between() needs two 3-vectors")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("angle_between() is undefined for a zero-length vector")
    cosine = float(np.dot(a, b)) / (norm_a * norm_b)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))
