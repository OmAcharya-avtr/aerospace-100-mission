"""TRIAD: the two-observation deterministic algorithm (Black 1964).

Algorithm
---------
Given a **primary** pair ``(b1, r1)`` and a **secondary** pair ``(b2, r2)``,
build an orthonormal triad in each frame:

    t1 = b1,   t2 = (b1 x b2) / |b1 x b2|,   t3 = t1 x t2                (Eq. T1)
    s1 = r1,   s2 = (r1 x r2) / |r1 x r2|,   s3 = s1 x s2                (Eq. T2)
    A = [t1 t2 t3] [s1 s2 s3]^T                                          (Eq. T3)

``A`` is orthogonal by construction and satisfies ``A s_k = t_k`` exactly, so
``A r1 = b1`` **exactly**: TRIAD trusts the primary observation completely and
uses the secondary one only for its component orthogonal to the primary.  It is
therefore not the Wahba optimum for any weights unless ``sigma_1 = 0``.

Frame order and conventions: see :mod:`wahbakit.conventions`.  Vectors are
dimensionless unit vectors; the result is the reference-to-body DCM ``A_BR``.

Near-parallel observations
--------------------------
Eq. T1 divides by ``|b1 x b2| = sin(eta)``.  As ``eta -> 0`` the second triad
axis is the normalised difference of two nearly equal vectors and loses
significance linearly in ``eta``; at ``eta = 0`` it is 0/0.  ``triad`` runs the
observability gate of Eq. O4 first and raises
:class:`~wahbakit.observations.DegenerateObservationsError` rather than
returning a plausible-looking but arbitrary matrix.  The error variance about
``b1`` grows as ``1 / sin^2(eta)`` (Eq. T4 in :mod:`wahbakit.covariance`), so
this is a real loss of information and not a numerical artefact.

References
----------
* H. D. Black, "A passive system for determining the attitude of a satellite",
  *AIAA Journal* **2**(7), 1350-1351 (1964).
* M. D. Shuster and S. D. Oh, *Journal of Guidance and Control* **4**(1), 70-77
  (1981).
* F. L. Markley, "Attitude determination using two vector measurements",
  NASA Goddard, Flight Mechanics Symposium (1999).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .observations import (
    DEFAULT_DEGENERACY_TOL,
    DegenerateObservationsError,
    VectorObservations,
)
from .solution import AttitudeSolution, build_solution

__all__ = ["triad", "triad_frame"]


def triad_frame(
    primary: NDArray[np.float64], secondary: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Orthonormal triad ``[t1 t2 t3]`` as columns, from Eq. T1.

    Parameters
    ----------
    primary, secondary : ndarray, shape (3,)
        Unit vectors in one frame.

    Raises
    ------
    DegenerateObservationsError
        If ``|primary x secondary| < 1e-12``.
    """
    t1 = np.asarray(primary, dtype=float).reshape(3)
    v = np.cross(t1, np.asarray(secondary, dtype=float).reshape(3))
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise DegenerateObservationsError(
            f"|primary x secondary| = {norm:.3e} < 1e-12: the two directions are "
            "parallel and Eq. T1 is 0/0"
        )
    t2 = v / norm
    return np.column_stack((t1, t2, np.cross(t1, t2)))


def triad(
    obs: VectorObservations,
    *,
    primary: int = 0,
    check_degeneracy: bool = True,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> AttitudeSolution:
    """Solve for attitude by TRIAD (Eq. T1-T3).

    Parameters
    ----------
    obs : VectorObservations
        Must hold exactly two observations.  TRIAD has no defined extension to
        more; select a pair with :meth:`VectorObservations.subset` and choose
        the most accurate sensor as ``primary``.
    primary : int, default 0
        Index of the observation that is reproduced exactly.  Set it to the
        smallest-``sigma`` sensor: the primary error enters the attitude
        one-for-one, the secondary error only through ``1 / sin(eta)``.
    check_degeneracy : bool, default True
        Run the Eq. O4 gate.  Setting it to ``False`` does not make the answer
        valid; it only removes the warning.
    degeneracy_tol : float, default 1e-6
        Gate on ``lambda_min`` of Eq. O4.

    Returns
    -------
    AttitudeSolution
        ``lambda_max`` is ``None`` (TRIAD never forms ``K``).  ``diagnostics``
        carries ``separation_deg`` (angle between the two body vectors),
        ``sin_separation`` (``|b1 x b2|``) and ``primary_index``.

    Raises
    ------
    ValueError
        If ``obs.n != 2`` or ``primary`` is not 0 or 1.
    DegenerateObservationsError
        If the geometry fails the Eq. O4 gate or the cross product vanishes.

    Notes
    -----
    Weights are ignored: TRIAD's answer depends only on which observation is
    primary.  The Wahba loss is still reported, computed with ``obs.weights``,
    so it can be compared with the optimal solvers.
    """
    if obs.n != 2:
        raise ValueError(
            f"TRIAD is defined for exactly 2 observations, got {obs.n}; "
            "use obs.subset([i, j]) to pick a pair, or use quest/q_method/olae"
        )
    if primary not in (0, 1):
        raise ValueError(f"primary must be 0 or 1, got {primary}")
    secondary = 1 - primary

    observability = obs.require_observable(degeneracy_tol) if check_degeneracy else None

    body = triad_frame(obs.body[primary], obs.body[secondary])
    ref = triad_frame(obs.reference[primary], obs.reference[secondary])
    dcm = body @ ref.T

    cross = float(np.linalg.norm(np.cross(obs.body[0], obs.body[1])))
    dot = float(np.dot(obs.body[0], obs.body[1]))
    return build_solution(
        dcm,
        "triad",
        obs,
        observability=observability,
        diagnostics={
            "separation_deg": float(np.degrees(np.arctan2(cross, dot))),
            "sin_separation": cross,
            "primary_index": float(primary),
        },
    )
