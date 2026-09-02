"""Singularity measure, classification and singular-surface mapping for SGCMG arrays.

Definitions used here
---------------------
For a Jacobian ``A(delta)`` of shape ``(3, n)`` with singular values
``sigma_1 >= sigma_2 >= sigma_3 >= 0``:

* **singularity measure** (Yoshikawa's manipulability, 1985)

      m(delta) = sqrt(det(A A^T)) = sigma_1 sigma_2 sigma_3   [(N*m*s/rad)^3]

  computed here as the product of the singular values, which is the same
  number and is numerically better behaved near ``m = 0``.
* **singular direction** ``u``: the left singular vector belonging to
  ``sigma_3``.  At a singular configuration ``u^T A = 0``, so no gimbal rate
  produces momentum rate along ``u`` and the array cannot deliver torque along
  ``u``.
* **sign vector** ``eps_i = sign(u . h_hat_i)``.  A singular configuration is
  **external** (a saturation singularity, on the boundary of the momentum
  envelope) when every ``eps_i`` has the same sign, and **internal**
  otherwise.  This is the classical classification of Margulies & Aubrun
  (1978), restated in Kurokawa's survey (2007).
* **passability**.  Along the singular direction, a gimbal perturbation that
  leaves ``h`` unchanged to first order changes ``h . u`` at second order by
  ``-(1/2) sum_i h0_i eps_i (d_delta_i)^2``.  Restricting that quadratic form
  to the null space of ``A`` gives a symmetric matrix ``Q``: definite ``Q``
  means the singularity is **elliptic** and null motion cannot escape it;
  indefinite ``Q`` means it is **hyperbolic** and null motion can.

Assumptions and validity: as in :mod:`cmgsteer.arrays`.  Everything below is
exact for the stated momentum map; no linearisation of the geometry is used
except where a second-order expansion is named explicitly (passability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arrays import CMGArray

__all__ = [
    "SingularityInfo",
    "classify_singularity",
    "condition_number",
    "fibonacci_directions",
    "manipulability_gradient",
    "momentum_envelope",
    "min_singular_value",
    "null_space_basis",
    "singular_configuration",
    "singular_direction",
    "singular_surface",
    "singularity_measure",
]

DEFAULT_SINGULAR_TOL = 1e-8


def _as_jacobian(a: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"Jacobian must have shape (3, n), got {arr.shape}")
    if arr.shape[1] < 3:
        raise ValueError(
            f"a 3-axis array needs at least 3 free gimbals, got {arr.shape[1]}; "
            "the array is under-actuated and every configuration is singular"
        )
    return arr


def singularity_measure(jacobian: ArrayLike) -> float:
    """``m = sqrt(det(A A^T))`` [(N*m*s/rad)^3], as the product of singular values."""
    sv = np.linalg.svd(_as_jacobian(jacobian), compute_uv=False)
    return float(np.prod(sv))


def min_singular_value(jacobian: ArrayLike) -> float:
    """Smallest singular value of ``A`` [N*m*s/rad]: the worst-direction torque gain."""
    return float(np.linalg.svd(_as_jacobian(jacobian), compute_uv=False)[-1])


def condition_number(jacobian: ArrayLike) -> float:
    """``sigma_max / sigma_min`` of ``A``; ``inf`` at an exactly singular configuration."""
    sv = np.linalg.svd(_as_jacobian(jacobian), compute_uv=False)
    return float(np.inf) if sv[-1] == 0.0 else float(sv[0] / sv[-1])


def singular_direction(jacobian: ArrayLike) -> NDArray[np.float64]:
    """Unit body-frame direction of least torque authority (left singular vector of sigma_min)."""
    u, _, _ = np.linalg.svd(_as_jacobian(jacobian))
    return np.ascontiguousarray(u[:, -1])


def null_space_basis(jacobian: ArrayLike, tol: float | None = None) -> NDArray[np.float64]:
    """Orthonormal basis of ``null(A)``, shape ``(n, n - rank)``.

    Columns are gimbal-rate directions that produce no momentum rate: the null
    motion the steering laws exploit.  ``tol`` is the absolute singular-value
    threshold for rank determination and defaults to
    ``max(A.shape) * eps * sigma_max``.
    """
    a = _as_jacobian(jacobian)
    u_, sv, vt = np.linalg.svd(a)
    del u_
    cutoff = tol if tol is not None else max(a.shape) * np.finfo(float).eps * sv[0]
    rank = int(np.count_nonzero(sv > cutoff))
    return np.ascontiguousarray(vt[rank:].T)


def manipulability_gradient(array: CMGArray, deltas: ArrayLike) -> NDArray[np.float64]:
    """Analytic gradient ``dm/ddelta`` over the free gimbals, shape ``(n_free,)``.

    Derived from ``m = prod_k sigma_k`` and the standard singular-value
    derivative ``dsigma_k = u_k^T (dA) v_k``.  Because ``dA/ddelta_j`` has only
    column ``j`` non-zero, equal to ``-h0_j h_hat_j``,

        dm/ddelta_j = sum_k (prod_{i != k} sigma_i) (u_k . a'_j) V[j, k]

    with ``a'_j = -h0_j h_hat_j``.  Written this way the expression contains no
    division, so it stays finite and accurate at ``m = 0`` where the
    determinant form ``adj(A A^T) / m`` is 0/0.

    Units: [(N*m*s/rad)^3 / rad].
    """
    d = np.asarray(deltas, dtype=float).reshape(-1)
    a = _as_jacobian(array.jacobian(d))
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    v = vt.T
    idx = array.free_indices
    # a'_j = d(A[:, j])/d(delta_j) = -h0_j * h_hat_j
    hhat = array.rotor_directions(d)[idx]
    aprime = -(array.rotor_momenta[idx][:, None] * hhat)  # (n_free, 3)
    # prod of the other singular values, without dividing by sigma_k
    cofactor = np.array([np.prod(np.delete(sv, k)) for k in range(sv.size)])
    #  (n_free, 3) @ (3, 3) -> (n_free, 3);  elementwise with V; weighted sum over k
    proj = aprime @ u  # proj[j, k] = u_k . a'_j
    return np.asarray((proj * v) @ cofactor, dtype=float)


def singular_configuration(
    array: CMGArray,
    direction: ArrayLike,
    signs: ArrayLike | None = None,
    tol: float = 1e-9,
) -> NDArray[np.float64]:
    """Gimbal angles of an analytically singular configuration.

    For a unit direction ``u`` and a sign vector ``eps in {-1, +1}^n``, the
    configuration in which every rotor momentum is as aligned with ``eps_i u``
    as its gimbal allows,

        h_hat_i = eps_i * normalise(u - (u . g_i) g_i),

    makes every Jacobian column ``a_i = h0_i (g_i x h_hat_i)`` perpendicular to
    ``u``, so ``u^T A = 0`` exactly and ``m = 0``.  This is the standard
    construction of the SGCMG singular set (Margulies & Aubrun 1978;
    Kurokawa 2007) and is what ``validation/validate_singularity.py`` checks
    the numerical singularity measure against.

    Parameters
    ----------
    array
        The CMG array.
    direction
        Body-frame direction ``u`` (normalised internally).
    signs
        Length-``n`` vector of ``+1``/``-1``.  ``None`` means all ``+1``, which
        gives the external (saturation) singularity in that direction.
    tol
        A direction within ``tol`` of a gimbal axis has no well-defined
        projection and raises ``ValueError``.

    Returns
    -------
    ``(n,)`` gimbal angles [rad] in ``(-pi, pi]``.
    """
    u = np.asarray(direction, dtype=float).reshape(-1)
    if u.shape != (3,):
        raise ValueError(f"direction must have shape (3,), got {u.shape}")
    norm = np.linalg.norm(u)
    if norm < tol:
        raise ValueError("direction must be a non-zero vector")
    u = u / norm

    n = array.n_cmgs
    eps = np.ones(n) if signs is None else np.asarray(signs, dtype=float).reshape(-1)
    if eps.shape[0] != n:
        raise ValueError(f"signs must have length {n}, got {eps.shape[0]}")
    if not np.all(np.isin(eps, (-1.0, 1.0))):
        raise ValueError("signs must contain only +1 and -1")

    g = array.gimbal_axes
    proj = u[None, :] - (g @ u)[:, None] * g
    lengths = np.linalg.norm(proj, axis=1)
    if np.any(lengths < tol):
        bad = int(np.argmin(lengths))
        raise ValueError(
            f"direction is parallel to gimbal axis {bad} ({array.names[bad]}); "
            "the singular configuration is undefined for that direction"
        )
    hhat = eps[:, None] * proj / lengths[:, None]
    c = array.ref_axes
    s = array.transverse_axes
    return np.arctan2(np.sum(hhat * s, axis=1), np.sum(hhat * c, axis=1))


@dataclass(frozen=True)
class SingularityInfo:
    """Classification of one configuration.

    Attributes
    ----------
    measure
        ``m = sqrt(det(A A^T))`` [(N*m*s/rad)^3].
    min_singular_value
        ``sigma_min`` of ``A`` [N*m*s/rad].
    condition_number
        ``sigma_max / sigma_min``.
    singular
        Whether ``sigma_min <= tol * sigma_max``.
    kind
        ``"none"``, ``"external"`` (saturation) or ``"internal"``.
    passability
        ``"none"``, ``"hyperbolic"`` (null motion can escape),
        ``"elliptic"`` (it cannot) or ``"degenerate"`` (the second-order form
        is itself singular, so second order does not decide).
    direction
        Unit body-frame singular direction ``u``.
    signs
        ``eps_i = sign(u . h_hat_i)`` over the free gimbals; ``0`` where the
        rotor momentum is perpendicular to ``u`` within ``tol``.
    momentum
        Array momentum ``h`` at this configuration [N*m*s].
    rank
        Numerical rank of ``A``.
    """

    measure: float
    min_singular_value: float
    condition_number: float
    singular: bool
    kind: str
    passability: str
    direction: NDArray[np.float64]
    signs: NDArray[np.float64]
    momentum: NDArray[np.float64]
    rank: int


def classify_singularity(
    array: CMGArray, deltas: ArrayLike, tol: float = DEFAULT_SINGULAR_TOL
) -> SingularityInfo:
    """Classify a configuration as regular, internal-singular or external-singular.

    ``tol`` is relative: the configuration counts as singular when
    ``sigma_min <= tol * sigma_max``.  The default 1e-8 is roughly the square
    root of double precision and is the value used throughout the validation
    scripts.
    """
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must lie in (0, 1), got {tol}")
    d = np.asarray(deltas, dtype=float).reshape(-1)
    a = _as_jacobian(array.jacobian(d))
    u_mat, sv, vt = np.linalg.svd(a)
    measure = float(np.prod(sv))
    sigma_min = float(sv[-1])
    cond = float(np.inf) if sigma_min == 0.0 else float(sv[0] / sigma_min)
    singular = sigma_min <= tol * sv[0]
    u = np.ascontiguousarray(u_mat[:, -1])
    idx = array.free_indices
    hhat = array.rotor_directions(d)[idx]
    dots = hhat @ u
    signs = np.where(np.abs(dots) < tol, 0.0, np.sign(dots))
    momentum = array.momentum(d)

    if not singular:
        return SingularityInfo(
            measure=measure,
            min_singular_value=sigma_min,
            condition_number=cond,
            singular=False,
            kind="none",
            passability="none",
            direction=u,
            signs=signs,
            momentum=momentum,
            rank=int(np.count_nonzero(sv > tol * sv[0])),
        )

    nonzero = signs[signs != 0.0]
    kind = "external" if nonzero.size and np.all(nonzero == nonzero[0]) else "internal"

    basis = null_space_basis(a, tol=tol * sv[0])
    if basis.shape[1] == 0:
        passability = "degenerate"
    else:
        weights = array.rotor_momenta[idx] * signs
        quad = basis.T @ (weights[:, None] * basis)
        eig = np.linalg.eigvalsh(0.5 * (quad + quad.T))
        scale = max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
        pos = np.any(eig > tol * scale)
        neg = np.any(eig < -tol * scale)
        if pos and neg:
            passability = "hyperbolic"
        elif np.any(np.abs(eig) <= tol * scale):
            passability = "degenerate"
        else:
            passability = "elliptic"

    return SingularityInfo(
        measure=measure,
        min_singular_value=sigma_min,
        condition_number=cond,
        singular=True,
        kind=kind,
        passability=passability,
        direction=u,
        signs=signs,
        momentum=momentum,
        rank=int(np.count_nonzero(sv > tol * sv[0])),
    )


def fibonacci_directions(n_points: int) -> NDArray[np.float64]:
    """``(n_points, 3)`` near-uniform unit vectors by the Fibonacci sphere construction."""
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    k = np.arange(n_points, dtype=float) + 0.5
    z = 1.0 - 2.0 * k / n_points
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + np.sqrt(5.0)) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def singular_surface(
    array: CMGArray,
    signs: ArrayLike | None = None,
    n_points: int = 2000,
    directions: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map one singular surface of the array.

    For a fixed sign vector, sweeping the singular direction ``u`` over the
    unit sphere sweeps out a two-dimensional surface in momentum space on
    which ``m = 0``.  ``signs = +1`` everywhere gives the outer momentum
    envelope; every other sign vector gives an internal singular surface.

    Returns
    -------
    ``(momenta, gimbal_angles)`` with shapes ``(k, 3)`` [N*m*s] and
    ``(k, n)`` [rad].  Directions parallel to a gimbal axis are dropped, so
    ``k`` can be smaller than the number of directions requested.
    """
    dirs = (
        fibonacci_directions(n_points)
        if directions is None
        else np.atleast_2d(np.asarray(directions, dtype=float))
    )
    momenta: list[NDArray[np.float64]] = []
    angles: list[NDArray[np.float64]] = []
    for u in dirs:
        try:
            d = singular_configuration(array, u, signs)
        except ValueError:
            continue
        angles.append(d)
        momenta.append(array.momentum(d))
    if not momenta:
        raise ValueError("no usable directions: every one was parallel to a gimbal axis")
    return np.array(momenta), np.array(angles)


def momentum_envelope(
    array: CMGArray, n_points: int = 2000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The external (saturation) singular surface: :func:`singular_surface` with all signs ``+1``.

    Every point on it is a momentum state the array can reach only by
    saturating in that direction, and at which it has no torque authority
    along the outward normal.
    """
    return singular_surface(array, signs=None, n_points=n_points)
