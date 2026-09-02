"""Closed-form threshold design and detection-delay expectations.

This module is the reason the package exists.  A residual test is trivial to
write and worthless without an answer to "what false-alarm rate did I just
buy, and how long will it take to see the fault?"  Every expression here has a
source.

Chi-squared test
----------------
For a consistent filter the normalised innovation squared
``eps_k = y_k^T S^-1 y_k`` is chi-squared with ``m`` degrees of freedom, and
the sum over a window of ``W`` independent samples is chi-squared with ``W m``
degrees of freedom (Bar-Shalom, Rong Li & Kirubarajan 2001, sec. 5.4, the
standard filter-consistency NIS test; Mehra & Peschon, *Automatica* 7(5),
1971, pp. 637-640, who introduced the innovation-based failure test).  So a
per-test false-alarm probability ``alpha`` is bought by the threshold

.. math::
    h = F^{-1}_{\\chi^2_{Wm}}(1 - \\alpha)

and nothing else has to be tuned.  Successive *overlapping* windows are not
independent, so the per-window ``alpha`` is not the rate of alarm *events*;
the package measures both and reports both.

CUSUM
-----
Page's cumulative-sum test (Page, *Biometrika* 41(1/2), 1954, pp. 100-115) in
its log-likelihood-ratio form for a mean shift ``0 -> mu`` in a unit-variance
Gaussian sequence ``p_k``:

.. math::
    s_k &= \\mu p_k - \\mu^2 / 2 \\\\
    g_k &= \\max(0,\\; g_{k-1} + s_k), \\qquad \\text{alarm when } g_k > h

The Kullback-Leibler information per sample is ``K = mu^2 / 2`` in both
directions.  Wald's approximation (Basseville and Nikiforov, *Detection of
Abrupt Changes: Theory and Application*, Prentice-Hall, 1993, ch. 2 and 5)
gives the mean run length by ignoring the boundary overshoot:

.. math::
    E_1[\\tau] \\approx h / K = 2h / \\mu^2

The Brownian-motion approximation with Siegmund's overshoot correction
(Siegmund, *Sequential Analysis: Tests and Confidence Intervals*, Springer,
1985) replaces the boundary ``h`` by ``h + 1.1652 \\sigma_s`` with
``sigma_s = mu`` the standard deviation of the increment, and gives

.. math::
    E_1[\\tau] &\\approx \\frac{2}{\\mu^2}\\left(e^{-b} + b - 1\\right) \\\\
    E_0[\\tau] &\\approx \\frac{2}{\\mu^2}\\left(e^{b} - b - 1\\right),
        \\qquad b = h + 1.1652\\,\\mu

``E_0`` is the mean time between false alarms, so ``1 / E_0`` is the
false-alarm rate per sample and is what :func:`cusum_threshold_for_arl0`
inverts.

**Run length or delay?**  Every expression in this module returns a *run
length*: the number of samples the test inspects, which is at least one.  The
*delay* reported by :func:`fdiscope.detectors.detection_delay` is an index
difference and is zero when the alarm lands on the onset sample itself, so

    run length = delay + 1

Getting this wrong is a systematic one-sample error, which is 3 % at a delay
of 30 samples and 50 % at a delay of one, and it was large enough to make the
validation in ``validation/cusum_delay.py`` disagree with the literature until
the convention was pinned down.  Compare like with like.

Steady-state residual under a sensor bias
-----------------------------------------
For a constant additive measurement bias ``b`` the mean estimation error
``e = x - x^-`` obeys, with no noise,

.. math::
    e_{k+1} = F(I - KH) e_k - F K b

whose fixed point is ``e_ss = [I - F(I - KH)]^{-1} (-F K b)``, giving a mean
innovation ``y_ss = H e_ss + b``.  This is the closed-form fault-to-residual
gain of model-based FDI (Chen and Patton 1999, ch. 3; Gertler 1998, ch. 5) and
it is what makes the CUSUM design non-circular: ``mu`` comes from the model,
not from the data the detector is then run on.

Units: thresholds and statistics are dimensionless; delays are in samples.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.stats import chi2, ncx2

from .kalman import KalmanFilter, steady_state_covariance

__all__ = [
    "SIEGMUND_RHO",
    "chi2_threshold",
    "chi2_false_alarm_rate",
    "chi2_detection_power",
    "cusum_kl_information",
    "cusum_delay_wald",
    "cusum_delay_siegmund",
    "cusum_arl0_siegmund",
    "cusum_threshold_for_arl0",
    "cusum_delay_mean_path",
    "steady_state_gain",
    "innovation_dc_gain",
    "steady_state_innovation_mean",
    "normalised_bias_signature",
]

#: Siegmund's two-boundary overshoot correction for a discrete random walk,
#: ``h -> h + SIEGMUND_RHO * sigma`` (Siegmund 1985).  The value 1.1652 is
#: twice the single-boundary constant 0.5826.
SIEGMUND_RHO: float = 1.1652


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
    return a


def _check_dof(dof: int) -> int:
    d = int(dof)
    if d < 1:
        raise ValueError(f"dof must be >= 1, got {dof!r}")
    return d


def _check_mu(mu: float) -> float:
    m = float(mu)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError(f"mu must be positive and finite, got {mu!r}")
    return m


def _check_threshold(h: float) -> float:
    v = float(h)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"threshold must be positive and finite, got {h!r}")
    return v


def chi2_threshold(alpha: float, dof: int) -> float:
    """Chi-squared threshold for a design false-alarm probability.

    Parameters
    ----------
    alpha : float
        Per-test false-alarm probability, strictly in ``(0, 1)``.
    dof : int
        Degrees of freedom: ``window_length * measurement_dimension``.

    Returns
    -------
    float
        ``h`` such that ``P(chi2_dof > h) = alpha``, dimensionless.

    Examples
    --------
    >>> round(chi2_threshold(0.05, 2), 6)
    5.991465
    """
    return float(chi2.isf(_check_alpha(alpha), _check_dof(dof)))


def chi2_false_alarm_rate(threshold: float, dof: int) -> float:
    """Design false-alarm probability of a chi-squared threshold.

    Inverse of :func:`chi2_threshold`.  Returns ``P(chi2_dof > threshold)``.
    """
    return float(chi2.sf(_check_threshold(threshold), _check_dof(dof)))


def chi2_detection_power(threshold: float, dof: int, noncentrality: float) -> float:
    """Probability that a *faulted* window exceeds the threshold.

    Under a residual mean shift the windowed NIS is non-central chi-squared
    with non-centrality ``lambda = W |mu|^2`` (Bar-Shalom et al. 2001,
    sec. 5.4).  Returns ``P(chi2'(dof, lambda) > threshold)``, dimensionless.

    Parameters
    ----------
    threshold : float
        Test threshold.
    dof : int
        Degrees of freedom.
    noncentrality : float
        ``lambda >= 0``.
    """
    lam = float(noncentrality)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"noncentrality must be >= 0 and finite, got {noncentrality!r}")
    return float(ncx2.sf(_check_threshold(threshold), _check_dof(dof), lam))


def cusum_kl_information(mu: float) -> float:
    """Kullback-Leibler information per sample, ``mu^2 / 2`` [nats]."""
    m = _check_mu(mu)
    return float(0.5 * m * m)


def cusum_delay_wald(threshold: float, mu: float) -> float:
    """Wald mean run length after the change, ``h / K`` [samples].

    Ignores both the boundary overshoot and the finite starting point.  The
    two errors have opposite signs and cancel at ``mu = 1 / 1.1652 = 0.858``:
    below that ``h / K`` over-predicts the run length, above it under-predicts.
    :func:`cusum_delay_siegmund` corrects both.

    Returns a run length, not an index-based delay; see the module docstring.
    """
    return float(_check_threshold(threshold) / cusum_kl_information(mu))


def cusum_delay_siegmund(threshold: float, mu: float) -> float:
    """Mean run length after the change, with Siegmund's correction [samples].

    Returns a run length, not an index-based delay; see the module docstring.
    """
    h = _check_threshold(threshold)
    m = _check_mu(mu)
    b = h + SIEGMUND_RHO * m
    return float((np.exp(-b) + b - 1.0) * 2.0 / (m * m))


def cusum_arl0_siegmund(threshold: float, mu: float) -> float:
    """Mean time between false alarms [samples], Siegmund approximation."""
    h = _check_threshold(threshold)
    m = _check_mu(mu)
    b = h + SIEGMUND_RHO * m
    return float((np.exp(b) - b - 1.0) * 2.0 / (m * m))


def cusum_threshold_for_arl0(arl0: float, mu: float) -> float:
    """Threshold giving a target mean time between false alarms.

    Parameters
    ----------
    arl0 : float
        Desired ``E_0[tau]`` in samples; must exceed 1.
    mu : float
        Design mean shift of the projected residual, dimensionless.

    Returns
    -------
    float
        Threshold ``h`` such that :func:`cusum_arl0_siegmund` returns
        ``arl0``.  Solved by Brent's method on ``[1e-9, 60]``.

    Raises
    ------
    ValueError
        If ``arl0 <= 1`` or the requested value is outside the bracket.
    """
    a = float(arl0)
    m = _check_mu(mu)
    if not np.isfinite(a) or a <= 1.0:
        raise ValueError(f"arl0 must be > 1, got {arl0!r}")

    def f(h: float) -> float:
        return cusum_arl0_siegmund(h, m) - a

    lo, hi = 1e-9, 60.0
    if f(lo) > 0.0:
        raise ValueError(
            f"arl0={a} is below the smallest achievable value "
            f"{cusum_arl0_siegmund(lo, m):.6g} for mu={m}"
        )
    if f(hi) < 0.0:
        raise ValueError(f"arl0={a} is above the largest achievable value for mu={m}")
    return float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-14))


def cusum_delay_mean_path(
    mean_profile: ArrayLike, mu: float, threshold: float
) -> float:
    """Run length of the noise-free CUSUM path [samples].

    The Wald and Siegmund expressions assume the projected residual steps
    straight from ``0`` to ``mu`` at onset.  In a closed loop it does not: the
    estimator absorbs part of the fault and the residual mean rises over the
    closed-loop error time constant.  This function runs the CUSUM recursion
    on the *deterministic* mean profile instead::

        g_k = max(0, g_{k-1} + mu * pbar_k - mu^2 / 2)

    and returns ``k + 1`` for the first ``k`` with ``g_k > h``, i.e. the number
    of samples inspected, matching the convention of the other expressions
    here.  It ignores fluctuation, so
    it is a prediction of the median-ish behaviour rather than of the mean,
    but it captures the part the closed-form expressions cannot: the rise
    time.

    Parameters
    ----------
    mean_profile : array_like, shape (N,)
        Mean of the projected residual from onset onwards, dimensionless.
        Obtain it from a noise-free run.
    mu : float
        Design mean shift of the CUSUM.
    threshold : float
        CUSUM threshold ``h``.

    Returns
    -------
    float
        Run length in samples (at least 1), or ``inf`` if the path never
        crosses.
    """
    profile = np.asarray(mean_profile, dtype=float).reshape(-1)
    if profile.size == 0:
        raise ValueError("mean_profile must be non-empty")
    if not np.all(np.isfinite(profile)):
        raise ValueError("mean_profile must be finite")
    m = _check_mu(mu)
    h = _check_threshold(threshold)
    acc = 0.0
    for k in range(profile.size):
        acc = max(0.0, acc + m * profile[k] - 0.5 * m * m)
        if acc > h:
            return float(k + 1)
    return float("inf")


def steady_state_gain(kf: KalmanFilter) -> NDArray[np.float64]:
    """Steady-state Kalman gain ``K`` of a time-invariant filter."""
    p_prior, s = steady_state_covariance(kf)
    return p_prior @ kf.h.T @ np.linalg.inv(s)


def innovation_dc_gain(kf: KalmanFilter) -> NDArray[np.float64]:
    """Matrix ``M`` mapping a constant measurement bias to the mean innovation.

    Substituting the fixed point of the mean error recursion gives
    ``y_ss = M b`` with

    .. math::
        M = [I - F(I - KH)]^{-1} (I - F)

    A zero column of ``M`` marks a bias direction that leaves no steady-state
    innovation at all, i.e. a fault that no residual test can detect once its
    transient has died.  For the double-integrator loop of
    :mod:`fdiscope.plant` the first column is exactly zero, so a constant
    *angle* bias is asymptotically undetectable.

    Returns
    -------
    ndarray, shape (m, m)
        ``M``, dimensionless when the measurement channels share units and a
        unit-conversion matrix otherwise.
    """
    k = steady_state_gain(kf)
    a_cl = kf.f @ (np.eye(kf.n) - k @ kf.h)
    return kf.h @ np.linalg.solve(np.eye(kf.n) - a_cl, np.eye(kf.n) - kf.f)


def steady_state_innovation_mean(
    kf: KalmanFilter, bias: ArrayLike
) -> NDArray[np.float64]:
    """Mean innovation under a constant additive measurement bias.

    Parameters
    ----------
    kf : KalmanFilter
        Time-invariant filter.
    bias : array_like, shape (m,)
        Constant bias added to the measurement, in measurement units.

    Returns
    -------
    ndarray, shape (m,)
        ``y_ss = H e_ss + b`` with
        ``e_ss = [I - F(I - KH)]^-1 (-F K b)`` (see module docstring).

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``I - F(I - KH)`` is singular, i.e. the closed-loop error dynamics
        have a unit eigenvalue and no steady state exists.
    """
    b = np.asarray(bias, dtype=float).reshape(-1)
    if b.size != kf.m:
        raise ValueError(f"bias must have {kf.m} elements, got {b.size}")
    k = steady_state_gain(kf)
    a_cl = kf.f @ (np.eye(kf.n) - k @ kf.h)
    e_ss = np.linalg.solve(np.eye(kf.n) - a_cl, -(kf.f @ k @ b))
    return kf.h @ e_ss + b


def normalised_bias_signature(
    kf: KalmanFilter, bias: ArrayLike
) -> tuple[NDArray[np.float64], float]:
    """Unit residual direction and mean shift for a constant sensor bias.

    Returns
    -------
    (direction, mu) : tuple
        ``direction`` is the unit vector of the steady-state *normalised*
        residual mean ``L^-1 y_ss``, and ``mu`` is its Euclidean norm --
        exactly the mean shift the scalar CUSUM of
        :func:`fdiscope.detectors.CusumDetector` is designed for.

    Raises
    ------
    ValueError
        If the steady-state residual mean is numerically zero, so no
        direction is defined.
    """
    _, s = steady_state_covariance(kf)
    chol = np.linalg.cholesky(s)
    y_ss = steady_state_innovation_mean(kf, bias)
    r_ss = np.linalg.solve(chol, y_ss)
    mu = float(np.linalg.norm(r_ss))
    if mu < 1e-12:
        raise ValueError("steady-state residual mean is zero; no signature direction")
    return r_ss / mu, mu
