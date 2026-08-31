"""Rotation conventions: quaternions, direction-cosine matrices, small-angle errors.

Every convention this package uses is fixed here and nowhere else.

Frames and the attitude matrix
------------------------------
An observation is a pair ``(b_i, r_i)`` of unit vectors: ``b_i`` measured in the
**body** frame, ``r_i`` known in the **reference** frame.  The attitude matrix
``A`` returned by every solver satisfies

    b_i ~= A r_i,

so ``A`` is the reference-to-body direction-cosine matrix, written ``A_BR``.
It is orthogonal with ``det(A) = +1``.  The frame order in the argument list is
always ``(body, reference)`` and never varies.

Quaternions
-----------
* **Scalar first**: ``q = [w, x, y, z]``.  This matches P007 QuatKit and is the
  opposite of ``scipy.spatial.transform.Rotation``, which stores ``[x, y, z, w]``.
* **Hamilton** product (``i j = k``); ``quat_multiply(q2, q1)`` composes the
  matrices as ``M(q2) M(q1)``.
* The quaternion-to-matrix map is

      M(q) = (w^2 - v.v) I + 2 v v^T + 2 w [v x]          (Eq. C1)

  with ``v = [x, y, z]`` and ``[v x]`` the cross-product matrix.  This is
  identical to ``Rotation.from_quat([x, y, z, w]).as_matrix()``.
* The returned attitude quaternion satisfies ``A = M(q)``.  Equivalently,
  ``Rotation.from_quat(np.roll(q, -1)).apply(r) == b``.
* ``q`` and ``-q`` are the same rotation.  Every public function returns the
  representative with ``w >= 0`` (:func:`quat_canonical`).

.. warning::
   Shuster's papers write the attitude matrix as
   ``A(q) = (q4^2 - q.q) I + 2 q q^T - 2 q4 [q x]`` with the scalar part
   **last**.  That is ``M(q)`` transposed.  The Davenport eigenvector is
   therefore converted in :mod:`wahbakit.davenport` before it is returned;
   see the note there.

Attitude error
--------------
The body-frame attitude error of an estimate ``A_est`` relative to truth
``A_true`` is the rotation vector

    delta_theta = log(A_est A_true^T)   [rad]                (Eq. C2)

so that ``A_est A_true^T = exp([delta_theta x])``.  All covariance matrices in
this package are ``E[delta_theta delta_theta^T]`` in rad^2, expressed in the
**body** frame.  This is the convention of Shuster & Oh (1981).

References
----------
* G. Wahba, "A least squares estimate of satellite attitude", *SIAM Review*
  **7**(3), 409 (1965).
* M. D. Shuster and S. D. Oh, "Three-axis attitude determination from vector
  observations", *Journal of Guidance and Control* **4**(1), 70-77 (1981).
* F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer (2014), Chapter 5.
* S. W. Shepperd, "Quaternion from rotation matrix", *Journal of Guidance and
  Control* **1**(3), 223-224 (1978).

Units: all angles in radians unless a name ends in ``_deg``.  Vectors are
dimensionless unit vectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ORTHOGONALITY_TOL",
    "UNIT_TOL",
    "angle_between_dcm",
    "attitude_error_vector",
    "dcm_from_quat",
    "is_rotation",
    "quat_canonical",
    "quat_conjugate",
    "quat_from_dcm",
    "quat_multiply",
    "quat_normalize",
    "rotation_vector_from_dcm",
    "skew",
    "unit_vectors",
]

#: Tolerance on ``|A^T A - I|`` and ``|det A - 1|`` used by :func:`is_rotation`.
ORTHOGONALITY_TOL = 1e-8

#: Tolerance on ``| |v| - 1 |`` used when unit vectors are required.
UNIT_TOL = 1e-6


def skew(v: ArrayLike) -> NDArray[np.float64]:
    """Cross-product matrix ``[v x]`` with ``[v x] u = v x u`` (dimensionless).

    Parameters
    ----------
    v : array_like, shape (3,)

    Returns
    -------
    ndarray, shape (3, 3)
    """
    a = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=float,
    )


def unit_vectors(v: ArrayLike, *, name: str = "vectors") -> NDArray[np.float64]:
    """Normalise an ``(N, 3)`` array of vectors to unit length.

    Raises
    ------
    ValueError
        If the array is not ``(N, 3)``, is not finite, or contains a vector of
        norm below ``1e-12`` (direction undefined).
    """
    a = np.atleast_2d(np.asarray(v, dtype=float))
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite entries")
    norms = np.linalg.norm(a, axis=1)
    bad = np.flatnonzero(norms < 1e-12)
    if bad.size:
        raise ValueError(
            f"{name} row(s) {bad.tolist()} have norm < 1e-12; direction is undefined"
        )
    return a / norms[:, None]


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    """Return ``q / |q|`` as a scalar-first unit quaternion.

    Raises
    ------
    ValueError
        If ``q`` is not length 4, not finite, or has norm below ``1e-12``.
    """
    a = np.asarray(q, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(f"quaternion must have 4 components [w, x, y, z], got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError("quaternion contains non-finite entries")
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("quaternion norm < 1e-12; cannot normalise")
    return a / n


def quat_canonical(q: ArrayLike) -> NDArray[np.float64]:
    """Return the representative of ``+-q`` with a non-negative scalar part.

    ``q`` and ``-q`` describe the same attitude; this removes the ambiguity so
    that two estimates can be compared componentwise.  When ``w == 0`` exactly
    (a rotation by pi) the sign of the first non-zero vector component is made
    positive instead.
    """
    a = quat_normalize(q)
    for value in a:
        if value > 0.0:
            return a
        if value < 0.0:
            return -a
    return a


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    """Conjugate ``[w, -x, -y, -z]``; for a unit quaternion this is the inverse."""
    a = np.asarray(q, dtype=float).reshape(4).copy()
    a[1:] *= -1.0
    return a


def quat_multiply(q2: ArrayLike, q1: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``q2 (x) q1``, scalar-first.

    Composition order matches the matrices: ``M(quat_multiply(q2, q1))`` equals
    ``M(q2) @ M(q1)``, i.e. apply ``q1`` first.
    """
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
    return np.array(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ],
        dtype=float,
    )


def dcm_from_quat(q: ArrayLike) -> NDArray[np.float64]:
    """Direction-cosine matrix ``M(q)`` from a scalar-first quaternion (Eq. C1).

    Parameters
    ----------
    q : array_like, shape (4,)
        ``[w, x, y, z]``.  Normalised internally.

    Returns
    -------
    ndarray, shape (3, 3)
        ``(w^2 - v.v) I + 2 v v^T + 2 w [v x]``.  Identical to
        ``scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix()``.
    """
    w, x, y, z = quat_normalize(q)
    v = np.array([x, y, z])
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew(v)


def quat_from_dcm(dcm: ArrayLike) -> NDArray[np.float64]:
    """Scalar-first quaternion from a rotation matrix, by Shepperd's method.

    Shepperd (1978) selects whichever of ``trace(A)``, ``A00``, ``A11``, ``A22``
    is largest so the divisor is never small; the worst-case divisor is
    ``sqrt(1)``.  The inverse of :func:`dcm_from_quat`, returned in canonical
    form (``w >= 0``).

    Raises
    ------
    ValueError
        If ``dcm`` is not (3, 3) or is not a rotation to within
        :data:`ORTHOGONALITY_TOL`.
    """
    a = np.asarray(dcm, dtype=float)
    if a.shape != (3, 3):
        raise ValueError(f"dcm must have shape (3, 3), got {a.shape}")
    if not is_rotation(a):
        raise ValueError(
            "dcm is not a proper rotation to within "
            f"{ORTHOGONALITY_TOL:g}: max|A^T A - I| = "
            f"{np.max(np.abs(a.T @ a - np.eye(3))):.3e}, det = {np.linalg.det(a):.6f}"
        )
    trace = a[0, 0] + a[1, 1] + a[2, 2]
    candidates = np.array([trace, a[0, 0], a[1, 1], a[2, 2]])
    k = int(np.argmax(candidates))
    if k == 0:
        s = np.sqrt(1.0 + trace)
        q = np.array(
            [s, (a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s, (a[1, 0] - a[0, 1]) / s]
        )
    elif k == 1:
        s = np.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2])
        q = np.array(
            [(a[2, 1] - a[1, 2]) / s, s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s]
        )
    elif k == 2:
        s = np.sqrt(1.0 - a[0, 0] + a[1, 1] - a[2, 2])
        q = np.array(
            [(a[0, 2] - a[2, 0]) / s, (a[0, 1] + a[1, 0]) / s, s, (a[1, 2] + a[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 - a[0, 0] - a[1, 1] + a[2, 2])
        q = np.array(
            [(a[1, 0] - a[0, 1]) / s, (a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, s]
        )
    return quat_canonical(0.5 * q)


def is_rotation(dcm: ArrayLike, tol: float = ORTHOGONALITY_TOL) -> bool:
    """True if ``dcm`` is orthogonal with ``det = +1`` to within ``tol``."""
    a = np.asarray(dcm, dtype=float)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        return False
    orthogonality = float(np.max(np.abs(a.T @ a - np.eye(3))))
    return orthogonality <= tol and abs(float(np.linalg.det(a)) - 1.0) <= tol


def rotation_vector_from_dcm(dcm: ArrayLike) -> NDArray[np.float64]:
    """Rotation vector ``log(A)`` in rad, with ``|log(A)|`` in ``[0, pi]``.

    Uses the quaternion route (never the ``arccos((tr - 1) / 2)`` form, which
    loses half its significant digits for small angles).
    """
    q = quat_canonical(quat_from_dcm(dcm))
    w = float(np.clip(q[0], -1.0, 1.0))
    v = q[1:]
    sin_half = float(np.linalg.norm(v))
    if sin_half < 1e-12:
        return 2.0 * v
    angle = 2.0 * np.arctan2(sin_half, w)
    return angle * v / sin_half


def attitude_error_vector(dcm_est: ArrayLike, dcm_true: ArrayLike) -> NDArray[np.float64]:
    """Body-frame attitude error ``log(A_est A_true^T)`` in rad (Eq. C2)."""
    a_est = np.asarray(dcm_est, dtype=float)
    a_true = np.asarray(dcm_true, dtype=float)
    return rotation_vector_from_dcm(a_est @ a_true.T)


def angle_between_dcm(dcm_a: ArrayLike, dcm_b: ArrayLike) -> float:
    """Magnitude of the rotation taking ``dcm_b`` to ``dcm_a``, in rad [0, pi]."""
    return float(np.linalg.norm(attitude_error_vector(dcm_a, dcm_b)))
