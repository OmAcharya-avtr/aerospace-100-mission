"""Attitude covariance from the measurement covariances.

All covariances are ``P = E[delta_theta delta_theta^T]`` in **rad^2**, with
``delta_theta = log(A_est A_true^T)`` the body-frame attitude error of Eq. C2.
They are first-order (small-error) results and are valid while
``sigma_i << 1 rad``; the Monte Carlo comparison in
``validation/validate_covariance.py`` measures where that stops being true.

Optimal estimator (q-method, QUEST)
-----------------------------------
Under the measurement model of Eq. O1 the Fisher information of the attitude is
Eq. O3, and the maximum-likelihood estimator attains its inverse:

    P_opt = [ sum_i sigma_i^-2 (I - b_i b_i^T) ]^-1     [rad^2]          (Eq. V1)

(Shuster 1978; Shuster & Oh 1981; Markley & Crassidis 2014, Ch. 5).  This is the
Cramer-Rao lower bound for Wahba's problem, so no estimator does better.  Note
that the *measured* body vectors are used: to first order this makes no
difference, and it is what is available at run time.  ``sum_i (I - b_i b_i^T)``
has trace ``2N`` and is singular whenever the ``b_i`` are all parallel, which is
the same degeneracy the Eq. O4 gate catches.

TRIAD
-----
TRIAD reproduces the primary observation exactly, so its error is the rotation
carrying the true body triad onto the measured one.  Writing ``c = b1 . b2``,
``s = |b1 x b2|`` and ``T = [t1 t2 t3]`` for the measured body triad of Eq. T1,
first-order propagation of Eq. O1 through Eq. T1 gives, in the triad basis,

                | (sigma_1^2 c^2 + sigma_2^2) / s^2    0             -sigma_1^2 c / s |
    P' =        |               0                  sigma_1^2                0         |
                |      -sigma_1^2 c / s                0            sigma_1^2         |

    P_TRIAD = T P' T^T                                  [rad^2]         (Eq. V2)

The three diagonal entries are the error variances about ``t1 = b1`` (the
primary direction, seen only through the secondary sensor and the geometry),
and about ``t2`` and ``t3``, each of which is just ``sigma_1^2`` because TRIAD
passes the primary measurement error straight through.  The ``1 / s^2 =
1 / sin^2(eta)`` growth is the quantitative form of "near-parallel observations
lose the rotation about the common axis".  The same result appears in
Shuster & Oh (1981) and Markley & Crassidis (2014, Ch. 5); the derivation is
reproduced in ``validation/VALIDATION.md`` and checked against Monte Carlo in
``validation/validate_covariance.py``.

``P_TRIAD - P_opt`` is positive semi-definite: TRIAD is never better than the
optimum, with equality only as ``sigma_1 -> 0``.  That inequality is one of the
checks in ``validation/validate_covariance.py``.

References
----------
* M. D. Shuster, AIAA-78-1249 (1978).
* M. D. Shuster and S. D. Oh, *Journal of Guidance and Control* **4**(1), 70-77
  (1981).
* F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer (2014), Chapter 5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .observations import (
    DEFAULT_DEGENERACY_TOL,
    DegenerateObservationsError,
    VectorObservations,
)
from .triad import triad_frame

__all__ = [
    "COVARIANCE_METHODS",
    "attitude_covariance",
    "covariance_axis_sigmas_deg",
    "optimal_covariance",
    "triad_covariance",
]

#: Method names accepted by :func:`attitude_covariance`.
COVARIANCE_METHODS = ("optimal", "q-method", "quest", "olae", "triad")


def optimal_covariance(
    obs: VectorObservations, *, degeneracy_tol: float = DEFAULT_DEGENERACY_TOL
) -> NDArray[np.float64]:
    """Cramer-Rao attitude covariance of Eq. V1, in rad^2, body frame.

    Parameters
    ----------
    obs : VectorObservations
        Must carry ``sigmas`` [rad].
    degeneracy_tol : float, default 1e-6
        Gate on Eq. O4; below it the inverse in Eq. V1 is meaningless.

    Returns
    -------
    ndarray, shape (3, 3), symmetric positive definite, units rad^2.

    Raises
    ------
    ValueError
        If ``sigmas`` were not supplied.
    DegenerateObservationsError
        If the geometry fails the Eq. O4 gate.
    """
    sigmas = obs.require_sigmas("attitude covariance")
    obs.require_observable(degeneracy_tol)
    inverse_variance = 1.0 / sigmas**2
    fisher = np.eye(3) * float(np.sum(inverse_variance)) - (
        obs.body * inverse_variance[:, None]
    ).T @ obs.body
    fisher = 0.5 * (fisher + fisher.T)
    condition = float(np.linalg.cond(fisher))
    if not np.isfinite(condition) or condition > 1e14:
        raise DegenerateObservationsError(
            f"the Fisher information of Eq. O3 has condition number {condition:.3e}: "
            "one axis carries essentially no information once the sigmas are applied, "
            "so Eq. V1 cannot be inverted meaningfully. Check obs.observability() and "
            "the spread of the sigmas."
        )
    return np.linalg.inv(fisher)


def triad_covariance(
    obs: VectorObservations,
    *,
    primary: int = 0,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> NDArray[np.float64]:
    """TRIAD attitude covariance of Eq. V2, in rad^2, body frame.

    Parameters
    ----------
    obs : VectorObservations
        Exactly two observations, carrying ``sigmas`` [rad].
    primary : int, default 0
        The observation TRIAD reproduces exactly.  Must match the ``primary``
        passed to :func:`wahbakit.triad.triad`; the covariance is not symmetric
        in the two observations.
    degeneracy_tol : float, default 1e-6

    Returns
    -------
    ndarray, shape (3, 3), units rad^2.
    """
    if obs.n != 2:
        raise ValueError(f"TRIAD covariance is defined for exactly 2 observations, got {obs.n}")
    if primary not in (0, 1):
        raise ValueError(f"primary must be 0 or 1, got {primary}")
    sigmas = obs.require_sigmas("TRIAD attitude covariance")
    obs.require_observable(degeneracy_tol)

    secondary = 1 - primary
    b1 = obs.body[primary]
    b2 = obs.body[secondary]
    var1 = float(sigmas[primary] ** 2)
    var2 = float(sigmas[secondary] ** 2)

    cos_eta = float(np.dot(b1, b2))
    sin_eta = float(np.linalg.norm(np.cross(b1, b2)))
    if sin_eta < 1e-12:
        raise DegenerateObservationsError(
            f"|b1 x b2| = {sin_eta:.3e}: Eq. V2 divides by sin(eta) and the TRIAD "
            "covariance about the primary direction is unbounded"
        )

    triad_basis = triad_frame(b1, b2)
    off_diagonal = -var1 * cos_eta / sin_eta
    p_local = np.array(
        [
            [(var1 * cos_eta**2 + var2) / sin_eta**2, 0.0, off_diagonal],
            [0.0, var1, 0.0],
            [off_diagonal, 0.0, var1],
        ]
    )
    return triad_basis @ p_local @ triad_basis.T


def attitude_covariance(
    obs: VectorObservations,
    method: str = "optimal",
    *,
    primary: int = 0,
    degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
) -> NDArray[np.float64]:
    """Attitude covariance in rad^2 for the named estimator.

    Parameters
    ----------
    obs : VectorObservations
        Must carry ``sigmas`` [rad].
    method : {"optimal", "q-method", "quest", "olae", "triad"}
        ``"optimal"``, ``"q-method"`` and ``"quest"`` all return Eq. V1, which
        those two estimators attain.  ``"olae"`` also returns Eq. V1, but as an
        **approximation**: OLAE minimises a reweighted cost and its covariance
        is slightly larger (measured at about 1.3 % in RMS error for the
        four-observation geometry in ``validation/validate_covariance.py``).
        ``"triad"`` returns Eq. V2.
    primary : int, default 0
        Only used by ``"triad"``.
    degeneracy_tol : float, default 1e-6

    Returns
    -------
    ndarray, shape (3, 3), units rad^2.

    Raises
    ------
    ValueError
        On an unknown method or missing sigmas.
    """
    if method not in COVARIANCE_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {COVARIANCE_METHODS}")
    if method == "triad":
        return triad_covariance(obs, primary=primary, degeneracy_tol=degeneracy_tol)
    return optimal_covariance(obs, degeneracy_tol=degeneracy_tol)


def covariance_axis_sigmas_deg(covariance: ArrayLike) -> NDArray[np.float64]:
    """Per-axis 1-sigma attitude errors in **degrees** from a rad^2 covariance.

    Returns ``sqrt(diag(P))`` converted to degrees; these are the body x, y, z
    components, not principal axes.  Use ``numpy.linalg.eigvalsh`` on ``P`` for
    the principal error ellipsoid.
    """
    p = np.asarray(covariance, dtype=float)
    if p.shape != (3, 3):
        raise ValueError(f"covariance must be (3, 3), got {p.shape}")
    return np.degrees(np.sqrt(np.diag(p)))
