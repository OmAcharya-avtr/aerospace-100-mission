"""OLAE: Optimal Linear Attitude Estimator (Mortari, Markley & Singla 2007).

OLAE parametrises the attitude by the Gibbs (Rodrigues) vector ``g`` through the
Cayley transform and turns the attitude problem into one linear system.  With
``A = dcm_from_quat([1, g] / sqrt(1 + g.g))`` (Eq. C1), the relation
``b = A r`` is *exactly* equivalent to the linear constraint

    d_i = -[s_i x] g,     d_i = b_i - r_i,   s_i = b_i + r_i              (Eq. L1)

which holds with no approximation for any rotation angle below pi.  Stacking
Eq. L1 over all observations and solving in weighted least squares gives the
3x3 normal equations

    M g = y,   M = sum_i w_i (|s_i|^2 I - s_i s_i^T),   y = sum_i w_i (s_i x d_i)
                                                                          (Eq. L2)

For unit vectors ``s_i x d_i = -2 (b_i x r_i)``, so ``y = -2 z`` with ``z`` the
Davenport vector of Eq. D2; ``M`` and ``y`` are formed from the general
expressions here so the code stays readable next to the derivation.

Optimality
----------
Eq. L1 is exact, so **noise-free data gives the exact attitude for any
weights**.  With noise, OLAE minimises
``sum_i w_i |d_i + [s_i x] g|^2``, and substituting Eq. L1 for the true ``g``
gives, for the per-observation residual ``e_i = b_i - A r_i``,

    |d_i + [s_i x] g|^2 = (1 + |g|^2) |e_i|^2 - (g . e_i)^2               (Eq. L3)

Both terms depend on ``g``, so OLAE is *not* algebraically identical to the
Wahba optimum: it minimises the same residuals under a rotation-dependent,
anisotropic reweighting.  The gradient of the OLAE cost at the Wahba optimum is
``O(sigma)``, not ``O(sigma^2)``, so **the two estimates differ at first order
in the measurement noise** and OLAE is slightly less efficient.  The size of
that difference is measured, not asserted: see
``validation/validate_agreement.py`` and ``validation/validate_covariance.py``
for the numbers on the geometries tested there.  The practical consequence is
that the analytic covariance of :mod:`wahbakit.covariance`, which is the
Cramer-Rao bound attained by the q-method and QUEST, is only an approximation
for OLAE.

Near-parallel observations and the 180-degree singularity
---------------------------------------------------------
Two failure modes, and they are different:

* **Degenerate geometry.** ``M`` in Eq. L2 becomes singular, exactly as the
  Fisher information of Eq. O4 does.  The Eq. O4 gate runs first;
  ``diagnostics["m_condition_number"]`` reports the 2-norm condition number of
  ``M`` regardless.
* **Rotation by pi.** ``|g| = tan(theta / 2) -> infinity``, which is a defect of
  the Gibbs parametrisation and not of the data.  ``sequential_rotation=True``
  (the default) solves Eq. L2 under the identity and the three 180-degree body
  rotations and keeps the candidate with the smallest ``|g|``, exactly as QUEST
  does for Eq. Q3.

References
----------
* D. Mortari, F. L. Markley and P. Singla, "Optimal Linear Attitude Estimator",
  *Journal of Guidance, Control, and Dynamics* **30**(6), 1619-1627 (2007).
* A. Cayley, "Sur quelques proprietes des determinants gauches", *Journal fur
  die reine und angewandte Mathematik* **32**, 119-123 (1846).
* F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer (2014), Chapter 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .conventions import dcm_from_quat, quat_canonical, quat_multiply, quat_normalize
from .observations import DEFAULT_DEGENERACY_TOL, VectorObservations
from .quest import SEQUENTIAL_ROTATION_QUATS
from .solution import AttitudeSolution, build_solution

__all__ = ["olae", "olae_normal_equations"]


def olae_normal_equations(
    body: NDArray[np.float64],
    reference: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Form ``(M, y)`` of Eq. L2.  Dimensionless.

    Parameters
    ----------
    body, reference : ndarray, shape (N, 3)
        Unit vectors, matched row for row.
    weights : ndarray, shape (N,)
    """
    s = body + reference
    d = body - reference
    s_sq = np.sum(s * s, axis=1)
    m = np.eye(3) * float(np.sum(weights * s_sq)) - (s * weights[:, None]).T @ s
    y = np.sum(weights[:, None] * np.cross(s, d), axis=0)
    return m, y


def olae(
    obs: VectorObservations,
    *,
    sequential_rotation: bool = True,
    check_degeneracy: bool = True,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> AttitudeSolution:
    """Solve for attitude by OLAE (Eq. L1-L2).

    Parameters
    ----------
    obs : VectorObservations
    sequential_rotation : bool, default True
        Solve Eq. L2 under the identity and the three 180-degree body rotations
        and keep the smallest ``|g|``.  Disable only to reproduce the bare
        Gibbs-vector solution, which diverges at a pi rotation.
    check_degeneracy : bool, default True
    degeneracy_tol : float, default 1e-6

    Returns
    -------
    AttitudeSolution
        ``lambda_max`` is ``None`` -- OLAE never forms ``K``.  ``diagnostics``
        carries ``gibbs_norm`` (``|g| = tan(theta/2)`` of the selected
        candidate), ``m_condition_number`` and ``sequential_rotation_index``.

    Raises
    ------
    DegenerateObservationsError
        If the geometry fails the Eq. O4 gate.
    RuntimeError
        If Eq. L2 is singular in every candidate frame.
    """
    observability = obs.require_observable(degeneracy_tol) if check_degeneracy else None
    candidates = SEQUENTIAL_ROTATION_QUATS if sequential_rotation else (
        SEQUENTIAL_ROTATION_QUATS[0],
    )

    best: tuple[float, NDArray[np.float64], float, int] | None = None
    condition = np.inf
    for index, q_pre in enumerate(candidates):
        rotated_reference = obs.reference @ dcm_from_quat(q_pre).T
        m, y = olae_normal_equations(obs.body, rotated_reference, obs.weights)
        condition = float(np.linalg.cond(m))
        if not np.isfinite(condition) or condition > 1e14:
            continue
        g = np.linalg.solve(m, y)
        gibbs_norm = float(np.linalg.norm(g))
        q_local = quat_normalize(np.array([1.0, g[0], g[1], g[2]]))
        q_total = quat_canonical(quat_multiply(q_local, q_pre))
        if best is None or gibbs_norm < best[0]:
            best = (gibbs_norm, q_total, condition, index)
    if best is None:
        raise RuntimeError(
            f"OLAE normal equations (Eq. L2) are singular in every candidate frame "
            f"(condition number {condition:.3e}); the observation geometry does not "
            "determine three axes. Inspect obs.observability()."
        )
    gibbs_norm, q, m_condition, index = best

    return build_solution(
        dcm_from_quat(q),
        "olae",
        obs,
        observability=observability,
        diagnostics={
            "gibbs_norm": gibbs_norm,
            "m_condition_number": m_condition,
            "sequential_rotation_index": float(index),
        },
        quaternion=q,
    )
