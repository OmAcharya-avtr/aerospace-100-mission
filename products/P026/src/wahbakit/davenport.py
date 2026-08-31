"""Davenport's q-method: the Wahba optimum as a 4x4 symmetric eigenproblem.

Davenport (1968) showed that the gain ``trace(A B^T)`` is the quadratic form
``q^T K q`` for a unit quaternion ``q``, so the Wahba optimum is the eigenvector
of ``K`` belonging to its largest eigenvalue:

    B = sum_i w_i b_i r_i^T,   sigma = trace(B),   S = B + B^T            (Eq. D1)
    z = [B23 - B32, B31 - B13, B12 - B21]^T = sum_i w_i (b_i x r_i)       (Eq. D2)

          | S - sigma I      z    |
    K =   |                       |   (4 x 4, symmetric)                  (Eq. D3)
          |     z^T       sigma   |

    K q_D = lambda_max q_D,   gain = lambda_max,   L = sum_i w_i - lambda_max

Dimensionless throughout.  Valid for any ``N >= 2`` with a non-degenerate
geometry; no small-angle assumption is made anywhere.

.. warning::
   **Quaternion convention.** ``K`` is written for a **scalar-last** quaternion
   ``q_D = [q1, q2, q3, q4]`` under Shuster's attitude matrix

       A(q_D) = (q4^2 - q.q) I + 2 q q^T - 2 q4 [q x],

   which is the transpose of :func:`wahbakit.conventions.dcm_from_quat`.  This
   module therefore returns ``q = [q4, -q1, -q2, -q3]``, canonicalised to
   ``w >= 0``, so that ``A = dcm_from_quat(q)`` with ``b_i ~= A r_i``.  The raw
   eigenvector is available in ``diagnostics`` only as its eigenvalue gap; if
   you need Shuster's ordering, take ``quat_conjugate(q)`` and roll the scalar
   to the end.

Near-parallel observations
--------------------------
When the geometry degenerates, the two largest eigenvalues of ``K`` coalesce and
the eigenvector belonging to ``lambda_max`` is an arbitrary member of a
two-dimensional invariant subspace: LAPACK returns *a* vector, with no
indication that it is one of infinitely many minimisers.  ``q_method`` runs the
Eq. O4 gate first and reports ``eigenvalue_gap = lambda_1 - lambda_2`` in
``diagnostics`` so the condition is visible even when the gate is disabled.

References
----------
* P. B. Davenport, "A vector approach to the algebra of rotations with
  applications", NASA TN D-4696 (1968).
* M. D. Shuster and S. D. Oh, *Journal of Guidance and Control* **4**(1), 70-77
  (1981).
* F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer (2014), Chapter 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import dcm_from_quat, quat_canonical
from .observations import DEFAULT_DEGENERACY_TOL, VectorObservations
from .solution import AttitudeSolution, build_solution

__all__ = ["davenport_matrix", "profile_parts", "q_method"]


def profile_parts(
    profile: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Split ``B`` into ``(S, z, sigma)`` of Eq. D1-D2.

    Returns
    -------
    S : ndarray, (3, 3)   ``B + B^T``, symmetric.
    z : ndarray, (3,)     ``[B23 - B32, B31 - B13, B12 - B21]``.
    sigma : float         ``trace(B)``.
    """
    b = np.asarray(profile, dtype=float)
    if b.shape != (3, 3):
        raise ValueError(f"profile matrix must be (3, 3), got {b.shape}")
    z = np.array([b[1, 2] - b[2, 1], b[2, 0] - b[0, 2], b[0, 1] - b[1, 0]])
    return b + b.T, z, float(np.trace(b))


def davenport_matrix(profile: ArrayLike) -> NDArray[np.float64]:
    """Davenport ``K`` of Eq. D3 (4, 4), scalar-last block ordering."""
    s, z, sigma = profile_parts(profile)
    k = np.empty((4, 4), dtype=float)
    k[:3, :3] = s - sigma * np.eye(3)
    k[:3, 3] = z
    k[3, :3] = z
    k[3, 3] = sigma
    return k


def _quaternion_from_davenport(q_d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shuster scalar-last eigenvector -> this package's scalar-first quaternion."""
    return quat_canonical(np.array([q_d[3], -q_d[0], -q_d[1], -q_d[2]]))


def q_method(
    obs: VectorObservations,
    *,
    check_degeneracy: bool = True,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> AttitudeSolution:
    """Solve Wahba's problem exactly by Davenport's q-method (Eq. D1-D3).

    This is the reference solution the other three methods are measured
    against: a symmetric 4x4 eigendecomposition has no iteration, no
    parametrisation singularity and no small-angle assumption.

    Parameters
    ----------
    obs : VectorObservations
        Two or more observations.
    check_degeneracy : bool, default True
        Run the Eq. O4 gate before solving.
    degeneracy_tol : float, default 1e-6

    Returns
    -------
    AttitudeSolution
        ``lambda_max`` is the largest eigenvalue of ``K``.  ``diagnostics``
        carries ``eigenvalue_gap`` (``lambda_1 - lambda_2``; small means the
        optimum is not isolated) and ``lambda_min``.

    Raises
    ------
    DegenerateObservationsError
        If the geometry fails the Eq. O4 gate.
    """
    observability = obs.require_observable(degeneracy_tol) if check_degeneracy else None
    k = davenport_matrix(obs.attitude_profile_matrix())
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    lambda_max = float(eigenvalues[-1])
    q = _quaternion_from_davenport(eigenvectors[:, -1])
    return build_solution(
        dcm_from_quat(q),
        "q-method",
        obs,
        observability=observability,
        lambda_max=lambda_max,
        diagnostics={
            "eigenvalue_gap": lambda_max - float(eigenvalues[-2]),
            "lambda_min": float(eigenvalues[0]),
        },
        quaternion=q,
    )
