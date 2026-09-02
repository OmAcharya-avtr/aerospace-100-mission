"""Vector geometry, frames and the attitude solve behind an identification.

Conventions, fixed once here and used everywhere in the package:

* A **direction** is a dimensionless unit 3-vector. Right ascension ``ra`` and
  declination ``dec`` are in radians in an equatorial inertial frame, and

  .. math:: \\hat{r} = (\\cos\\delta\\cos\\alpha,\; \\cos\\delta\\sin\\alpha,\;
            \\sin\\delta)                                          (Eq. G1)

* An **attitude** is a direction cosine matrix ``A`` mapping the inertial
  (catalogue) frame to the camera frame: ``v_cam = A @ r_inertial``. The camera
  boresight is ``+z`` in the camera frame.
* A **quaternion** is scalar first, ``q = [w, x, y, z]``, Hamilton convention,
  with ``dcm_from_quat`` matching ``scipy.spatial.transform.Rotation
  .from_quat([x, y, z, w]).as_matrix()``. SciPy is not a runtime dependency;
  the agreement is checked in ``validation/validate_geometry.py``.

Angular separation uses

.. math:: \\theta = \\operatorname{atan2}(\\|a\\times b\\|,\; a\\cdot b)  (Eq. G2)

rather than ``arccos(a.b)``. The two agree analytically; ``arccos`` loses about
half its significant digits for small ``theta`` because ``a.b`` approaches 1
there, which matters directly here — the whole package is built on differences
of angles at the 1e-5 rad level.

The attitude solve is Davenport's q-method for Wahba's problem (Wahba 1965;
Davenport, NASA TN D-4696, 1968):

.. math::
    B = \\sum_i w_i\\, b_i r_i^T,\\quad
    K = \\begin{bmatrix} B + B^T - \\sigma I & z \\\\ z^T & \\sigma\\end{bmatrix},
    \\quad \\sigma = \\operatorname{tr} B,\;
    z = [B_{23}-B_{32}, B_{31}-B_{13}, B_{12}-B_{21}]^T              (Eq. G3)

with the optimal quaternion the eigenvector of ``K`` of largest eigenvalue.
This package reimplements it rather than importing a sibling product, so the
repository stands alone; the conventions match P007 QuatKit and P026 WahbaKit.

Validity: everything here is pure rotation geometry with no small-angle
approximation, valid for any rotation. Units are radians throughout unless a
name ends in ``_deg`` or ``_arcsec``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ARCSEC",
    "angle_between_dcm",
    "angular_separation",
    "davenport_attitude",
    "dcm_from_quat",
    "normalise",
    "quat_from_dcm",
    "radec_from_unit_vectors",
    "random_rotation",
    "skew",
    "unit_vectors_from_radec",
]

#: Radians in one arcsecond.
ARCSEC = np.pi / (180.0 * 3600.0)


def _as_vectors(v: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (N, 3), got {np.shape(v)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def normalise(v: ArrayLike, name: str = "vectors") -> np.ndarray:
    """Return ``v`` scaled to unit length, shape ``(N, 3)`` [dimensionless].

    Raises ``ValueError`` on a zero-length or non-finite row: a direction with
    no direction is a caller bug, not something to paper over with ``nan``.
    """
    arr = _as_vectors(v, name)
    norms = np.linalg.norm(arr, axis=1)
    bad = norms <= 0.0
    if np.any(bad):
        raise ValueError(f"{name} contains {int(bad.sum())} zero-length row(s)")
    return arr / norms[:, None]


def unit_vectors_from_radec(ra_rad: ArrayLike, dec_rad: ArrayLike) -> np.ndarray:
    """Eq. G1. ``ra_rad``, ``dec_rad`` [rad] -> unit vectors ``(N, 3)``."""
    ra = np.atleast_1d(np.asarray(ra_rad, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_rad, dtype=float))
    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec must have the same shape, got {ra.shape} and {dec.shape}")
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        raise ValueError("ra and dec must be finite")
    if np.any(np.abs(dec) > np.pi / 2 + 1e-12):
        raise ValueError("dec must lie in [-pi/2, pi/2] rad")
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=1)


def radec_from_unit_vectors(v: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of Eq. G1. Returns ``(ra, dec)`` [rad], ``ra`` in ``[0, 2*pi)``."""
    arr = normalise(v, "vectors")
    ra = np.mod(np.arctan2(arr[:, 1], arr[:, 0]), 2.0 * np.pi)
    dec = np.arcsin(np.clip(arr[:, 2], -1.0, 1.0))
    return ra, dec


def angular_separation(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Eq. G2. Angle [rad] between rows of ``a`` and ``b``, broadcast over rows.

    Inputs need not be normalised; they are normalised on entry.
    """
    av = normalise(a, "a")
    bv = normalise(b, "b")
    if av.shape[0] != bv.shape[0] and av.shape[0] != 1 and bv.shape[0] != 1:
        raise ValueError(f"cannot broadcast shapes {av.shape} and {bv.shape}")
    cross = np.linalg.norm(np.cross(av, bv), axis=-1)
    dot = np.sum(av * bv, axis=-1)
    return np.arctan2(cross, dot)


def skew(v: ArrayLike) -> np.ndarray:
    """Cross-product matrix ``[v x]``, shape ``(3, 3)``."""
    w = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def dcm_from_quat(q: ArrayLike) -> np.ndarray:
    """Scalar-first unit quaternion -> DCM ``(3, 3)``.

    ``A = (w^2 - v.v) I + 2 v v^T + 2 w [v x]``, identical to SciPy's
    ``Rotation.from_quat([x, y, z, w]).as_matrix()``.
    """
    qq = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(qq)
    if n <= 0.0 or not np.isfinite(n):
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = qq / n
    v = np.array([x, y, z])
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew(v)


def quat_from_dcm(a: ArrayLike) -> np.ndarray:
    """DCM -> scalar-first quaternion with ``w >= 0`` (Shepperd 1978, 4 branches)."""
    m = np.asarray(a, dtype=float).reshape(3, 3)
    if not np.allclose(m @ m.T, np.eye(3), atol=1e-8) or np.linalg.det(m) < 0.0:
        raise ValueError("matrix is not a proper rotation (A A^T = I, det A = +1)")
    tr = np.trace(m)
    candidates = np.array([tr, m[0, 0], m[1, 1], m[2, 2]])
    branch = int(np.argmax(candidates))
    if branch == 0:
        w = 0.5 * np.sqrt(1.0 + tr)
        s = 0.25 / w
        q = np.array([w, s * (m[2, 1] - m[1, 2]), s * (m[0, 2] - m[2, 0]), s * (m[1, 0] - m[0, 1])])
    else:
        i = branch - 1
        j, k = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        s = 0.5 / r
        q = np.zeros(4)
        q[0] = s * (m[k, j] - m[j, k])
        q[1 + i] = 0.5 * r
        q[1 + j] = s * (m[j, i] + m[i, j])
        q[1 + k] = s * (m[k, i] + m[i, k])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A rotation drawn from the uniform (Haar) measure on SO(3).

    Uses Shoemake's (1992) subgroup algorithm for a uniform unit quaternion,
    which is uniform on SO(3) because the quaternion double cover is a Haar
    isomorphism.
    """
    u1, u2, u3 = rng.random(3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    q = np.array(
        [
            s2 * np.cos(2.0 * np.pi * u3),
            s1 * np.sin(2.0 * np.pi * u2),
            s1 * np.cos(2.0 * np.pi * u2),
            s2 * np.sin(2.0 * np.pi * u3),
        ]
    )
    return dcm_from_quat(q)


def davenport_attitude(
    body: ArrayLike, reference: ArrayLike, weights: ArrayLike | None = None
) -> np.ndarray:
    """Eq. G3. Optimal DCM ``A`` with ``body_i ~= A @ reference_i``.

    Parameters
    ----------
    body, reference
        ``(N, 3)`` unit vectors, ``N >= 2``. ``body`` is in the camera frame,
        ``reference`` in the catalogue frame.
    weights
        ``(N,)`` non-negative weights [dimensionless]; equal weights if omitted.

    Returns the ``(3, 3)`` DCM. Raises ``ValueError`` if fewer than two
    observations are given, or if the observations are collinear enough that
    the largest two eigenvalues of ``K`` are degenerate — in that case the
    rotation about the common axis is not in the data and the answer would be
    invented rather than estimated.
    """
    b = normalise(body, "body")
    r = normalise(reference, "reference")
    if b.shape != r.shape:
        raise ValueError(f"body {b.shape} and reference {r.shape} must have the same shape")
    if b.shape[0] < 2:
        raise ValueError(f"need at least 2 observations for an attitude, got {b.shape[0]}")
    if weights is None:
        w = np.full(b.shape[0], 1.0 / b.shape[0])
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != b.shape[0]:
            raise ValueError(f"weights has length {w.shape[0]}, expected {b.shape[0]}")
        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
        total = w.sum()
        if total <= 0.0:
            raise ValueError("weights must not all be zero")
        w = w / total

    bmat = (w[:, None] * b).T @ r
    sigma = np.trace(bmat)
    s = bmat + bmat.T
    z = np.array(
        [bmat[1, 2] - bmat[2, 1], bmat[2, 0] - bmat[0, 2], bmat[0, 1] - bmat[1, 0]]
    )
    k = np.zeros((4, 4))
    k[:3, :3] = s - sigma * np.eye(3)
    k[:3, 3] = z
    k[3, :3] = z
    k[3, 3] = sigma
    evals, evecs = np.linalg.eigh(k)
    if evals[-1] - evals[-2] <= 1e-12:
        raise ValueError(
            "attitude is not observable from these observations: the two largest "
            f"eigenvalues of K differ by {evals[-1] - evals[-2]:.3e} <= 1e-12 (Eq. G3). "
            "The directions are collinear to within numerical precision."
        )
    v = evecs[:, -1]
    # K is built in Shuster's convention, whose attitude matrix is
    # (w^2 - v.v) I + 2 v v^T - 2 w [v x] -- the TRANSPOSE of the map this
    # package uses (and of scipy's).  Negating the vector part converts
    # between them.  Without this the returned matrix is A^T: still
    # orthogonal, still det +1, and silently the inverse rotation.  There is
    # an explicit regression test for it in tests/test_geometry.py.
    q = np.array([v[3], -v[0], -v[1], -v[2]])
    return dcm_from_quat(q)


def angle_between_dcm(a: ArrayLike, b: ArrayLike) -> float:
    """Rotation angle [rad] of ``A B^T``, in ``[0, pi]``.

    Computed through the quaternion as ``2 atan2(|v|, w)`` rather than from
    ``arccos((tr M - 1)/2)``. The trace form is analytically identical and
    numerically useless near zero: ``tr M`` approaches 3, and ``arccos(1 - d)``
    grows as ``sqrt(2d)``, so a double-precision trace caps the resolvable
    angle at about ``3e-8`` rad. That floor is not the estimator's error, and
    a test that used the trace form would report it as if it were
    (``validation/validate_geometry.py`` section 1d).
    """
    m = np.asarray(a, dtype=float).reshape(3, 3) @ np.asarray(b, dtype=float).reshape(3, 3).T
    q = quat_from_dcm(m)
    return float(2.0 * np.arctan2(float(np.linalg.norm(q[1:])), float(q[0])))
