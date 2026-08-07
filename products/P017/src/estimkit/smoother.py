r"""Rauch-Tung-Striebel fixed-interval smoother.

Given the forward filter's prior and posterior sequences, the RTS
backward recursion produces the fixed-interval smoothed estimates
:math:`\hat x_{k|T}` -- each state conditioned on **all** ``T``
measurements, not only those up to ``k``:

.. math::
    A_k &= P^{+}_{k} F_{k+1}^{\mathsf{T}} (P^{-}_{k+1})^{-1} \\
    \hat x_{k|T} &= \hat x^{+}_{k}
        + A_k\left(\hat x_{k+1|T} - \hat x^{-}_{k+1}\right) \\
    P_{k|T} &= P^{+}_{k}
        + A_k\left(P_{k+1|T} - P^{-}_{k+1}\right)A_k^{\mathsf{T}}

initialised at the last step with
:math:`\hat x_{T-1|T} = \hat x^{+}_{T-1}`,
:math:`P_{T-1|T} = P^{+}_{T-1}`.

References
----------
- Rauch, H. E., Tung, F. and Striebel, C. T., "Maximum likelihood
  estimates of linear dynamic systems", *AIAA Journal*, Vol. 3, No. 8,
  1965 -- the original derivation.
- Bar-Shalom, Rong Li & Kirubarajan, *Estimation with Applications to
  Tracking and Navigation*, Wiley 2001, Ch. 6 (smoothing).
- Simon, D., *Optimal State Estimation*, Wiley 2006, Ch. 9 (optimal
  smoothing).
- Sarkka, S., *Bayesian Filtering and Smoothing*, Cambridge 2013, Ch. 8 --
  the extended and unscented RTS variants used here for the nonlinear
  filters.

Guaranteed property
-------------------
For a correctly specified linear-Gaussian model the smoothed covariance
satisfies :math:`P_{k|T} \preceq P^{+}_{k}` (the difference is negative
semi-definite), because the smoother conditions on strictly more data.
Smoothing therefore reduces the expected squared error at every interior
step; the reduction is largest in the middle of the interval and vanishes
at the final step, where filter and smoother coincide by construction.
This is checked numerically in ``validation/``.

Applicability to nonlinear filters
----------------------------------
The same recursion is applied to EKF output using the per-step transition
Jacobian, and to UKF output using the sigma-point cross-covariance recast
as an effective transition matrix (see
:meth:`estimkit.ukf.UnscentedKalmanFilter.filter`). Both are
approximations: the covariance-ordering guarantee above holds exactly only
in the linear-Gaussian case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .covariance import symmetrize
from .linear import FilterResult

__all__ = ["SmootherResult", "rts_smooth"]


@dataclass
class SmootherResult:
    """Fixed-interval smoothed means, covariances and smoother gains.

    Attributes
    ----------
    x : ndarray, shape (T, n)
        Smoothed state means :math:`\\hat x_{k|T}`.
    p : ndarray, shape (T, n, n)
        Smoothed covariances :math:`P_{k|T}`, symmetrised.
    gain : ndarray, shape (T, n, n)
        Smoother gains :math:`A_k`. The last entry is zero (unused).
    """

    x: NDArray[np.float64]
    p: NDArray[np.float64]
    gain: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.x.shape[0])


def rts_smooth(
    result: FilterResult | None = None,
    *,
    x_prior: ArrayLike | None = None,
    p_prior: ArrayLike | None = None,
    x_post: ArrayLike | None = None,
    p_post: ArrayLike | None = None,
    transition: ArrayLike | None = None,
) -> SmootherResult:
    """Run the RTS backward recursion.

    Parameters
    ----------
    result : FilterResult, optional
        Output of :meth:`estimkit.linear.KalmanFilter.filter`,
        :meth:`estimkit.ekf.ExtendedKalmanFilter.filter` or
        :meth:`estimkit.ukf.UnscentedKalmanFilter.filter`. If given, the
        keyword arguments are ignored.
    x_prior : array_like, shape (T, n)
        Predicted means :math:`\\hat x^{-}_{k}`.
    p_prior : array_like, shape (T, n, n)
        Predicted covariances :math:`P^{-}_{k}`.
    x_post : array_like, shape (T, n)
        Filtered means :math:`\\hat x^{+}_{k}`.
    p_post : array_like, shape (T, n, n)
        Filtered covariances :math:`P^{+}_{k}`.
    transition : array_like, shape (n, n) or (T, n, n)
        Transition matrix (or per-step matrices). ``transition[k]`` is the
        matrix that maps step ``k-1`` to step ``k``, matching the
        convention of :class:`estimkit.linear.FilterResult`.

    Returns
    -------
    SmootherResult

    Raises
    ------
    ValueError
        On inconsistent shapes, fewer than one time step, or a singular
        predicted covariance (which itself indicates covariance collapse in
        the forward pass).
    """
    if result is not None:
        x_prior = result.x_prior
        p_prior = result.p_prior
        x_post = result.x_post
        p_post = result.p_post
        transition = result.transition
    missing = [
        name
        for name, val in (
            ("x_prior", x_prior),
            ("p_prior", p_prior),
            ("x_post", x_post),
            ("p_post", p_post),
            ("transition", transition),
        )
        if val is None
    ]
    if missing:
        raise ValueError(
            "rts_smooth needs either a FilterResult or all of "
            f"x_prior/p_prior/x_post/p_post/transition; missing: {', '.join(missing)}"
        )

    xm = np.asarray(x_prior, dtype=float)
    pm = np.asarray(p_prior, dtype=float)
    xp = np.asarray(x_post, dtype=float)
    pp = np.asarray(p_post, dtype=float)
    if xm.ndim != 2 or xp.shape != xm.shape:
        raise ValueError(f"x_prior and x_post must both be (T, n); got {xm.shape}, {xp.shape}")
    t, n = xm.shape
    if t < 1:
        raise ValueError("need at least one time step")
    if pm.shape != (t, n, n) or pp.shape != (t, n, n):
        raise ValueError(
            f"p_prior and p_post must be ({t}, {n}, {n}); got {pm.shape}, {pp.shape}"
        )
    fm = np.asarray(transition, dtype=float)
    if fm.ndim == 2:
        if fm.shape != (n, n):
            raise ValueError(f"transition must be ({n}, {n}); got {fm.shape}")
        fm = np.repeat(fm[None, :, :], t, axis=0)
    elif fm.shape != (t, n, n):
        raise ValueError(f"transition must be ({n}, {n}) or ({t}, {n}, {n}); got {fm.shape}")

    xs = np.empty_like(xp)
    ps = np.empty_like(pp)
    gains = np.zeros((t, n, n))
    xs[-1] = xp[-1]
    ps[-1] = symmetrize(pp[-1])

    for k in range(t - 2, -1, -1):
        try:
            # A_k = P^+_k F_{k+1}^T (P^-_{k+1})^{-1}, solved rather than inverted.
            a = np.linalg.solve(symmetrize(pm[k + 1]), fm[k + 1] @ symmetrize(pp[k])).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"predicted covariance at step {k + 1} is singular; the forward filter "
                "has collapsed (see estimkit.covariance.covariance_health)"
            ) from exc
        gains[k] = a
        xs[k] = xp[k] + a @ (xs[k + 1] - xm[k + 1])
        ps[k] = symmetrize(pp[k] + a @ (ps[k + 1] - pm[k + 1]) @ a.T)

    return SmootherResult(xs, ps, gains)
