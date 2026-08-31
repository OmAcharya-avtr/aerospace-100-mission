"""QUEST: the Wahba optimum without an eigendecomposition (Shuster & Oh 1981).

QUEST finds the same ``lambda_max`` as Davenport's ``K`` (Eq. D3) as a root of
the quartic characteristic polynomial, then writes the eigenvector in closed
form.  With ``S``, ``z``, ``sigma`` from Eq. D1-D2:

    kappa = trace(adj S) = (trace(S)^2 - trace(S^2)) / 2,   Delta = det S  (Eq. Q1)
    a = sigma^2 - kappa
    b = sigma^2 + z.z
    c = Delta + z^T S z
    d = z^T S^2 z
    psi(lambda) = lambda^4 - (a + b) lambda^2 - c lambda + (a b + c sigma - d) = 0
                                                                            (Eq. Q2)

Eq. Q2 is the characteristic polynomial of ``K`` written out; its largest root
is ``lambda_max``.  Because the weights sum to one, ``lambda_max <= 1`` with
equality for noise-free data, and ``lambda_0 = 1`` is an excellent Newton start:
the deficit ``1 - lambda_max`` is the Wahba loss, of order ``sigma^2``, so one
or two iterations reach machine precision for realistic sensor noise.

The eigenvector follows from the adjugate of ``(lambda + sigma) I - S``
(Shuster & Oh 1981), which never requires a matrix inverse:

    alpha = lambda^2 - sigma^2 + kappa
    beta  = lambda - sigma
    gamma = (lambda + sigma) alpha - Delta
    X     = (alpha I + beta S + S^2) z                                     (Eq. Q3)
    q_D   = [X; gamma] / sqrt(|X|^2 + gamma^2)      (Shuster, scalar last)

and is converted to this package's scalar-first convention exactly as in
:mod:`wahbakit.davenport`: ``q = [gamma, -X] / norm``, canonicalised to
``w >= 0``.  Dimensionless throughout; no small-angle assumption.

The 180-degree singularity and the method of sequential rotations
------------------------------------------------------------------
``gamma`` and ``X`` vanish together when the attitude is a rotation by exactly
pi, because the scalar part of the quaternion is then zero and Eq. Q3 is 0/0.
Shuster's remedy, implemented here as ``sequential_rotation=True`` (the
default), is to pre-rotate the reference vectors by each of the three
180-degree rotations about the body axes, solve Eq. Q3 four times, and keep the
candidate with the largest scalar part -- the best-conditioned of the four.
The composition ``q = q' (x) q_k`` is exact, so this costs four 3x3 evaluations
and changes nothing else.  ``lambda_max`` is invariant under the pre-rotation
and is computed once.

Near-parallel observations
--------------------------
Degenerate geometry makes ``lambda_max`` a near-double root of Eq. Q2, so
``psi'(lambda) -> 0`` and Newton's iteration loses its quadratic rate; the
returned attitude is then one arbitrary member of a continuum.  ``quest`` runs
the Eq. O4 gate first and additionally reports ``newton_iterations`` and
``characteristic_residual`` so the condition is visible.

References
----------
* M. D. Shuster and S. D. Oh, "Three-axis attitude determination from vector
  observations", *Journal of Guidance and Control* **4**(1), 70-77 (1981).
* M. D. Shuster, "Approximate algorithms for fast optimal attitude
  computation", AIAA-78-1249 (1978).
* F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer (2014), Chapter 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import dcm_from_quat, quat_canonical, quat_multiply
from .davenport import profile_parts
from .observations import DEFAULT_DEGENERACY_TOL, VectorObservations
from .solution import AttitudeSolution, build_solution

__all__ = [
    "SEQUENTIAL_ROTATION_QUATS",
    "STALL_STEP",
    "characteristic_coefficients",
    "characteristic_polynomial",
    "quest",
    "quest_lambda_max",
]

#: Identity plus the three 180-degree rotations about the body axes, scalar
#: first.  Used for Shuster's method of sequential rotations.
STALL_STEP = 1e-9

SEQUENTIAL_ROTATION_QUATS: tuple[NDArray[np.float64], ...] = (
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0]),
)


def characteristic_coefficients(profile: ArrayLike) -> dict[str, float]:
    """Coefficients of Eq. Q1-Q2 from the attitude profile matrix ``B``.

    Returns
    -------
    dict
        Keys ``sigma``, ``kappa``, ``delta``, ``a``, ``b``, ``c``, ``d``.  All
        dimensionless.
    """
    s, z, sigma = profile_parts(profile)
    kappa = 0.5 * (float(np.trace(s)) ** 2 - float(np.trace(s @ s)))
    delta = float(np.linalg.det(s))
    return {
        "sigma": sigma,
        "kappa": kappa,
        "delta": delta,
        "a": sigma**2 - kappa,
        "b": sigma**2 + float(z @ z),
        "c": delta + float(z @ s @ z),
        "d": float(z @ (s @ s) @ z),
    }


def characteristic_polynomial(lam: float, coefficients: dict[str, float]) -> float:
    """Evaluate ``psi(lambda)`` of Eq. Q2.  Dimensionless."""
    a = coefficients["a"]
    b = coefficients["b"]
    c = coefficients["c"]
    d = coefficients["d"]
    sigma = coefficients["sigma"]
    return lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * sigma - d)


def quest_lambda_max(
    coefficients: dict[str, float],
    lambda0: float = 1.0,
    *,
    tol: float = 1e-14,
    max_iter: int = 50,
) -> tuple[float, int, float]:
    """Newton-Raphson root of Eq. Q2 starting from ``lambda0``.

    Parameters
    ----------
    coefficients : dict
        From :func:`characteristic_coefficients`.
    lambda0 : float, default 1.0
        Start value.  ``sum_i w_i = 1`` is the standard choice and is an upper
        bound on ``lambda_max``.
    tol : float, default 1e-14
        Convergence test on ``|lambda_{k+1} - lambda_k|``.  The iteration also
        stops when the step stops shrinking while already below ``STALL_STEP``
        = 1e-9, which is round-off stagnation rather than failure.
    max_iter : int, default 50

    Returns
    -------
    lambda_max : float
    iterations : int
    residual : float
        ``|psi(lambda_max)|``, which should be at round-off.

    Raises
    ------
    RuntimeError
        If the iteration does not converge, or the derivative vanishes (a
        double root, i.e. a degenerate geometry).  Use
        :func:`wahbakit.davenport.q_method`, which does not iterate.
    """
    a = coefficients["a"]
    b = coefficients["b"]
    c = coefficients["c"]
    lam = float(lambda0)
    step = np.inf
    previous = np.inf
    for iteration in range(1, max_iter + 1):
        psi = characteristic_polynomial(lam, coefficients)
        dpsi = 4.0 * lam**3 - 2.0 * (a + b) * lam - c
        if abs(dpsi) < 1e-14:
            raise RuntimeError(
                f"QUEST characteristic polynomial has psi'(lambda) = {dpsi:.3e} at "
                f"lambda = {lam:.12f}: lambda_max is a near-double root, i.e. the "
                "observation geometry is degenerate. Use q_method, which does not "
                "iterate, and inspect obs.observability()."
            )
        step = psi / dpsi
        lam -= step
        if abs(step) < tol or (abs(step) >= previous and abs(step) < STALL_STEP):
            return lam, iteration, abs(characteristic_polynomial(lam, coefficients))
        previous = abs(step)
    raise RuntimeError(
        f"QUEST Newton iteration did not converge in {max_iter} steps "
        f"(last step {abs(step):.3e}, tol {tol:.3e}). Use q_method."
    )


def _closed_form_quaternion(
    s: NDArray[np.float64], z: NDArray[np.float64], sigma: float, lam: float
) -> tuple[NDArray[np.float64], float, float] | None:
    """Eq. Q3.  Returns (quaternion scalar-first, gamma, |X|), or None if 0/0."""
    kappa = 0.5 * (float(np.trace(s)) ** 2 - float(np.trace(s @ s)))
    delta = float(np.linalg.det(s))
    alpha = lam**2 - sigma**2 + kappa
    beta = lam - sigma
    gamma = (lam + sigma) * alpha - delta
    x = (alpha * np.eye(3) + beta * s + s @ s) @ z
    norm = float(np.hypot(np.linalg.norm(x), gamma))
    if norm < 1e-14 * max(1.0, abs(lam)):
        return None
    return np.array([gamma, -x[0], -x[1], -x[2]]) / norm, gamma, float(np.linalg.norm(x))


def quest(
    obs: VectorObservations,
    *,
    sequential_rotation: bool = True,
    newton_tol: float = 1e-14,
    max_iter: int = 50,
    check_degeneracy: bool = True,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> AttitudeSolution:
    """Solve Wahba's problem by QUEST (Eq. Q1-Q3).

    Parameters
    ----------
    obs : VectorObservations
    sequential_rotation : bool, default True
        Evaluate Eq. Q3 under the identity and the three 180-degree body
        rotations and keep the best-conditioned candidate.  Disable only to
        reproduce the bare Eq. Q3, which fails at a pi rotation.
    newton_tol, max_iter : float, int
        Passed to :func:`quest_lambda_max`.
    check_degeneracy : bool, default True
    degeneracy_tol : float, default 1e-6

    Returns
    -------
    AttitudeSolution
        ``lambda_max`` is the root of Eq. Q2.  ``diagnostics`` carries
        ``newton_iterations``, ``characteristic_residual`` (``|psi|`` at the
        root), ``gamma`` and ``x_norm`` from Eq. Q3 for the selected candidate,
        and ``sequential_rotation_index`` (0 = identity, 1/2/3 = 180 deg about
        body x/y/z).

    Raises
    ------
    DegenerateObservationsError
        If the geometry fails the Eq. O4 gate.
    RuntimeError
        If Newton fails to converge or Eq. Q3 is 0/0 in every frame.
    """
    observability = obs.require_observable(degeneracy_tol) if check_degeneracy else None
    profile = obs.attitude_profile_matrix()
    coefficients = characteristic_coefficients(profile)
    lam, iterations, residual = quest_lambda_max(
        coefficients, float(np.sum(obs.weights)), tol=newton_tol, max_iter=max_iter
    )

    candidates = SEQUENTIAL_ROTATION_QUATS if sequential_rotation else (
        SEQUENTIAL_ROTATION_QUATS[0],
    )
    best: tuple[float, NDArray[np.float64], float, float, int] | None = None
    for index, q_pre in enumerate(candidates):
        rotation = dcm_from_quat(q_pre)
        s_k, z_k, sigma_k = profile_parts(profile @ rotation.T)
        candidate = _closed_form_quaternion(s_k, z_k, sigma_k, lam)
        if candidate is None:  # Eq. Q3 is 0/0 in this frame; try the next one
            continue
        q_k, gamma, x_norm = candidate
        scalar = abs(q_k[0])
        if best is None or scalar > best[0]:
            best = (scalar, quat_canonical(quat_multiply(q_k, q_pre)), gamma, x_norm, index)
    if best is None:
        raise RuntimeError(
            "QUEST Eq. Q3 gives gamma = 0 and X = 0 in every candidate frame; the "
            "attitude is a rotation by exactly pi and the closed form is 0/0. Use "
            "q_method, which has no parametrisation singularity."
        )
    _, q, gamma, x_norm, index = best

    return build_solution(
        dcm_from_quat(q),
        "quest",
        obs,
        observability=observability,
        lambda_max=lam,
        diagnostics={
            "newton_iterations": float(iterations),
            "characteristic_residual": residual,
            "gamma": gamma,
            "x_norm": x_norm,
            "sequential_rotation_index": float(index),
        },
        quaternion=q,
    )
