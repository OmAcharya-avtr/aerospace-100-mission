"""One entry point for all four solvers, plus the method-selection guidance.

Which method to use
-------------------

============  ================================================================
Method        Use it when
============  ================================================================
``q-method``  You want the answer and do not care about the cost.  A 4x4
              symmetric eigendecomposition is exact, non-iterative, has no
              parametrisation singularity, and is the reference every other
              method here is measured against.  Any ``N >= 2``.
``quest``     You want the same answer without an eigensolver: one quartic
              Newton root (typically 2-4 iterations from ``lambda_0 = 1``) plus
              a closed form.  Agrees with ``q-method`` to 1.1e-11 rad over the
              500 random problems in ``validation/validate_agreement.py``.
``olae``      You want a single 3x3 linear solve and can accept a slightly
              less efficient estimator: OLAE minimises a reweighted cost and
              its estimate differs from the Wahba optimum at first order in the
              measurement noise (about 1.3 % larger RMS error on the geometry
              measured in ``validation/``).
``triad``     You have exactly two observations of very different quality and
              want the more accurate one reproduced exactly, or you want a
              deterministic answer with no eigenvalue and no iteration at all.
              It is not optimal for any weights unless ``sigma_1 = 0``.
============  ================================================================

Every method here returns the same :class:`~wahbakit.solution.AttitudeSolution`
with the same conventions: ``b_i ~= A r_i``, scalar-first quaternion with
``w >= 0``, ``A = dcm_from_quat(q)``.
"""

from __future__ import annotations

from collections.abc import Callable

from .covariance import attitude_covariance
from .davenport import q_method
from .observations import DEFAULT_DEGENERACY_TOL, VectorObservations
from .olae import olae
from .quest import quest
from .solution import AttitudeSolution
from .triad import triad

__all__ = ["METHODS", "solve_wahba"]

#: Method name -> solver.  ``"davenport"`` is an alias for ``"q-method"``.
METHODS: dict[str, Callable[..., AttitudeSolution]] = {
    "triad": triad,
    "q-method": q_method,
    "davenport": q_method,
    "quest": quest,
    "olae": olae,
}


def solve_wahba(
    obs: VectorObservations,
    method: str = "quest",
    *,
    check_degeneracy: bool = True,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
    with_covariance: bool = False,
    **kwargs: object,
) -> AttitudeSolution | tuple[AttitudeSolution, object]:
    """Solve Wahba's problem by the named method.

    Parameters
    ----------
    obs : VectorObservations
    method : {"triad", "q-method", "davenport", "quest", "olae"}, default "quest"
    check_degeneracy : bool, default True
        Run the Eq. O4 gate before solving.
    degeneracy_tol : float, default 1e-6
    with_covariance : bool, default False
        Also return the attitude covariance in rad^2 for this method
        (:func:`wahbakit.covariance.attitude_covariance`).  Requires ``obs`` to
        carry ``sigmas``.
    **kwargs
        Forwarded to the solver: ``primary`` for TRIAD, ``sequential_rotation``
        for QUEST and OLAE, ``newton_tol`` and ``max_iter`` for QUEST.

    Returns
    -------
    AttitudeSolution, or (AttitudeSolution, ndarray) if ``with_covariance``.

    Raises
    ------
    ValueError
        On an unknown method name.
    DegenerateObservationsError
        If the geometry fails the Eq. O4 gate.
    """
    if method not in METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(METHODS)}"
        )
    solution = METHODS[method](
        obs, check_degeneracy=check_degeneracy, degeneracy_tol=degeneracy_tol, **kwargs
    )
    if not with_covariance:
        return solution
    covariance_method = "q-method" if method == "davenport" else method
    covariance = attitude_covariance(
        obs,
        covariance_method,
        primary=int(kwargs.get("primary", 0)),  # type: ignore[call-overload]
        degeneracy_tol=degeneracy_tol,
    )
    return solution, covariance
