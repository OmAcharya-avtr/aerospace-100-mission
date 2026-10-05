"""Average run length and window false-alarm probability for CUSUM, EWMA and OOL.

All three detectors in this module are designed, not tuned: a target operating
point is named first and the threshold that delivers it is solved for.  Two
operating-point currencies are used.

``ARL`` (average run length)
    The textbook quantity: the expected number of samples until the chart
    alarms.  ``ARL0`` is the in-control value, ``ARL1`` the out-of-control one.

``alpha_W`` (window false-alarm probability)
    The probability that a chart started from its reset state alarms at least
    once within ``W`` samples of in-control data.  This is the currency used for
    the matched-false-alarm-rate comparison in :mod:`telemetryool.harness`,
    because it is a Bernoulli trial per window and therefore has an exact
    binomial standard error.

Model assumptions, common to every function here
------------------------------------------------
* Observations are independent and identically distributed
  ``Normal(mu0 + delta * sigma, sigma^2)``; ``delta`` is the mean shift in
  units of ``sigma`` (dimensionless).
* ``sigma`` is known.  Charts operate on the standardised deviate
  ``(x - mu0) / sigma``, so every threshold below is in sigma units.
* Serial correlation is **not** modelled.  Under AR(1) telemetry the designed
  ``alpha_W`` is not delivered; the size of that error is measured in
  ``validation/validate_far_design.py`` rather than assumed away.

References
----------
Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* 41(1/2),
    100-115.  Origin of the cumulative-sum scheme.
Brook, D. and Evans, D. A. (1972). "An approach to the probability distribution
    of CUSUM run length." *Biometrika* 59(3), 539-549.  The Markov-chain
    discretisation used by :func:`cusum_arl_markov`.
Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence Intervals*.
    Springer.  The closed-form CUSUM ARL approximation used by
    :func:`cusum_arl_siegmund`.
Lucas, J. M. and Saccucci, M. S. (1990). "Exponentially Weighted Moving Average
    Control Schemes: Properties and Enhancements." *Technometrics* 32(1), 1-12.
    The Markov-chain method used by :func:`ewma_arl_markov`.
Montgomery, D. C. (2013). *Introduction to Statistical Quality Control*,
    7th ed., Wiley.  Chapter 9 collects the CUSUM and EWMA design procedures,
    including the Siegmund approximation with ``b = h + 1.166``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

__all__ = [
    "SIEGMUND_B_OFFSET",
    "ChartDesign",
    "cusum_arl_siegmund",
    "cusum_arl_siegmund_two_sided",
    "cusum_arl_markov",
    "cusum_window_false_alarm",
    "ewma_arl_markov",
    "ewma_window_false_alarm",
    "ool_window_false_alarm",
    "ool_arl",
    "design_cusum_h",
    "design_ewma_L",
    "design_ool_limit",
]

#: Correction term in the Siegmund CUSUM ARL approximation, ``b = h + 1.166``.
#: Siegmund (1985); tabulated in Montgomery (2013) ch. 9.  Dimensionless.
SIEGMUND_B_OFFSET = 1.166


@dataclass(frozen=True)
class ChartDesign:
    """Result of a threshold design.

    Attributes
    ----------
    threshold
        The designed decision interval ``h`` (CUSUM, sigma units), control-limit
        multiplier ``L`` (EWMA, dimensionless) or limit multiplier ``L`` (OOL,
        dimensionless).
    target_alpha_w
        Requested window false-alarm probability (dimensionless, in (0, 1)).
    achieved_alpha_w
        Window false-alarm probability the returned threshold delivers under the
        iid-normal model, from the same numerical method used to design it.
    window_length
        ``W``, samples per monitoring window.
    arl0
        In-control average run length at the returned threshold, samples.
    method
        Name of the numerical method that produced ``achieved_alpha_w``.
    """

    threshold: float
    target_alpha_w: float
    achieved_alpha_w: float
    window_length: int
    arl0: float
    method: str


def _check_prob(name: str, value: float) -> float:
    v = float(value)
    if not (0.0 < v < 1.0):
        raise ValueError(f"{name} must lie strictly in (0, 1), got {value!r}")
    return v


def _check_window(window_length: int) -> int:
    if not isinstance(window_length, (int, np.integer)) or window_length < 1:
        raise ValueError(f"window_length must be an integer >= 1, got {window_length!r}")
    return int(window_length)


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #


def cusum_arl_siegmund(delta: float, k: float, h: float) -> float:
    """One-sided upper-CUSUM ARL by the Siegmund (1985) closed-form approximation.

    ``ARL = (exp(-2 * Delta * b) + 2 * Delta * b - 1) / (2 * Delta**2)`` with
    ``Delta = delta - k`` and ``b = h + 1.166``; the removable singularity at
    ``Delta = 0`` has the limit ``ARL = b**2``.  Near that singularity the
    expression is evaluated from its Maclaurin series instead, because the
    direct form cancels catastrophically.

    Parameters
    ----------
    delta
        True mean shift in sigma units (dimensionless).  ``0`` is in-control.
    k
        CUSUM reference value ("allowance") in sigma units, > 0.  The
        conventional choice for detecting a shift ``delta*`` is ``k = delta*/2``.
    h
        Decision interval in sigma units, > 0.

    Returns
    -------
    float
        Expected number of samples until the upper arm exceeds ``h``.

    Notes
    -----
    Valid for the iid-normal model with known sigma.  Siegmund's derivation is
    an asymptotic (large-``h``) corrected-boundary result; accuracy degrades for
    small ``h`` and for ``|delta - k|`` close to zero.  Its error against the
    Brook-Evans chain is quantified in ``validation/validate_cusum_delay.py``.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k!r}")
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h!r}")
    b = float(h) + SIEGMUND_B_OFFSET
    d = float(delta) - float(k)
    x = 2.0 * d * b
    if abs(x) < 1.0e-3:
        # The direct expression loses all its significant digits to cancellation
        # as x -> 0, because the numerator is O(x^2) while its terms are O(1).
        # Substituting the Maclaurin series of exp(-x) gives
        # ARL = b^2 (1 - x/3 + x^2/12 - x^3/60 + O(x^4)), whose truncation error
        # at |x| = 1e-3 is below 2e-11 relative.
        return float(b * b * (1.0 - x / 3.0 + x * x / 12.0 - x**3 / 60.0))
    return float((np.exp(-x) + x - 1.0) / (2.0 * d * d))


def cusum_arl_siegmund_two_sided(delta: float, k: float, h: float) -> float:
    """Two-sided CUSUM ARL from the Siegmund approximation on each arm.

    ``1 / ARL = 1 / ARL(delta) + 1 / ARL(-delta)`` with each arm from
    :func:`cusum_arl_siegmund` -- the competing-risks combination recommended in
    Montgomery (2013) ch. 9.  For ``delta = 0`` this is simply half the
    one-sided value.
    """
    up = cusum_arl_siegmund(delta, k, h)
    dn = cusum_arl_siegmund(-float(delta), k, h)
    return float(1.0 / (1.0 / up + 1.0 / dn))


def _cusum_chain(delta: float, k: float, h: float, n_states: int):
    """Brook-Evans transition matrix for the upper CUSUM arm.

    ``[0, h)`` is split into ``n_states`` intervals of width ``w = h / n_states``;
    interval ``j`` (0-based) is ``[j*w, (j+1)*w)`` with midpoint ``(j + 0.5) * w``.
    Interval 0 contains the reset value ``C = 0``, so a reset maps into it.

    Returns ``(P, start_row)`` where ``P[i, j]`` is the one-step probability of
    moving from the midpoint of interval ``i`` into interval ``j``, and
    ``start_row[j]`` is the same probability starting from exactly ``C = 0``.
    """
    w = float(h) / n_states
    mids = (np.arange(n_states) + 0.5) * w
    edges = np.arange(n_states + 1) * w  # 0, w, ..., h

    def rows(origins: NDArray[np.float64]) -> NDArray[np.float64]:
        # C' = max(0, origin + x - k), x ~ N(delta, 1).
        # P(C' < edge) = Phi(edge - origin + k - delta) for edge > 0.
        arg = edges[None, :] - origins[:, None] + float(k) - float(delta)
        cdf = norm.cdf(arg)
        out = np.diff(cdf, axis=1)
        # All mass with origin + x - k < 0 resets to C = 0, which lies in
        # interval 0, so it belongs in column 0.  cdf[:, 0] = P(C' < 0).
        out[:, 0] += cdf[:, 0]
        return out

    return rows(mids), rows(np.zeros(1))[0]


def cusum_arl_markov(delta: float, k: float, h: float, n_states: int = 400) -> float:
    """Two-sided CUSUM ARL by the Brook-Evans (1972) Markov-chain discretisation.

    Parameters
    ----------
    delta
        Mean shift in sigma units (dimensionless).
    k
        Reference value in sigma units, > 0.
    h
        Decision interval in sigma units, > 0.
    n_states
        Number of discretisation intervals on ``[0, h)`` per arm.  Error is
        ``O(1 / n_states)``; 400 is the package default and its residual
        discretisation error is measured in
        ``validation/validate_cusum_delay.py``.

    Returns
    -------
    float
        Expected samples until either arm exceeds ``h``.

    Notes
    -----
    The two arms are combined as ``1 / ARL = 1 / ARL_plus + 1 / ARL_minus``,
    which treats the arms as independent competing risks.  That is the standard
    approximation (Montgomery 2013 ch. 9); the arms are in fact negatively
    dependent through the shared observation, so the combined value is slightly
    conservative.  Its size is measured against Monte Carlo in
    ``validation/validate_far_design.py``.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states!r}")
    arl_up = _cusum_arm_arl(float(delta), k, h, n_states)
    arl_dn = _cusum_arm_arl(-float(delta), k, h, n_states)
    return float(1.0 / (1.0 / arl_up + 1.0 / arl_dn))


def _cusum_arm_arl(delta: float, k: float, h: float, n_states: int) -> float:
    p_mat, start = _cusum_chain(delta, k, h, n_states)
    mu = np.linalg.solve(np.eye(n_states) - p_mat, np.ones(n_states))
    return float(1.0 + start @ mu)


def _survival(p_mat, start, window_length: int) -> float:
    """Probability of surviving ``window_length`` steps without absorption."""
    v = start.copy()
    for _ in range(window_length - 1):
        v = v @ p_mat
    return float(v.sum())


def cusum_window_false_alarm(
    k: float, h: float, window_length: int, n_states: int = 200, delta: float = 0.0
) -> float:
    """Probability that a two-sided CUSUM alarms within ``window_length`` samples.

    Parameters
    ----------
    k, h
        Reference value and decision interval, sigma units.
    window_length
        ``W``, samples in the monitoring window, >= 1.
    n_states
        Brook-Evans discretisation intervals per arm.
    delta
        Mean shift in sigma units; ``0`` gives the false-alarm probability and a
        non-zero value gives the detection probability within ``W``.

    Returns
    -------
    float
        ``1 - P(no alarm in W samples)``, dimensionless.

    Notes
    -----
    The two arms are propagated independently and their survivals multiplied,
    the same competing-risks approximation as :func:`cusum_arl_markov`.
    """
    window_length = _check_window(window_length)
    s = 1.0
    for d in (float(delta), -float(delta)):
        p_mat, start = _cusum_chain(d, k, h, n_states)
        s *= _survival(p_mat, start, window_length)
    return float(1.0 - s)


def design_cusum_h(
    target_alpha_w: float,
    k: float,
    window_length: int,
    n_states: int = 200,
    bracket: tuple[float, float] = (0.05, 60.0),
    tol: float = 1e-10,
) -> ChartDesign:
    """Solve for the CUSUM decision interval ``h`` delivering ``target_alpha_w``.

    Bisection on the monotone map ``h -> alpha_W(h)`` from
    :func:`cusum_window_false_alarm`.  Raises ``ValueError`` if the target lies
    outside the bracket's reachable range.
    """
    target = _check_prob("target_alpha_w", target_alpha_w)
    window_length = _check_window(window_length)
    lo, hi = bracket

    def f(h: float) -> float:
        return cusum_window_false_alarm(k, h, window_length, n_states) - target

    if f(lo) < 0 or f(hi) > 0:
        raise ValueError(
            f"target_alpha_w={target} is not reachable for k={k}, W={window_length} "
            f"within h in {bracket}: alpha_W spans "
            f"[{f(hi) + target:.3e}, {f(lo) + target:.3e}]"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    h = 0.5 * (lo + hi)
    return ChartDesign(
        threshold=h,
        target_alpha_w=target,
        achieved_alpha_w=cusum_window_false_alarm(k, h, window_length, n_states),
        window_length=window_length,
        arl0=cusum_arl_markov(0.0, k, h, max(n_states, 400)),
        method=f"brook-evans markov chain, n_states={n_states}",
    )


# --------------------------------------------------------------------------- #
# EWMA
# --------------------------------------------------------------------------- #


def ewma_sigma_z(lam: float) -> float:
    """Steady-state standard deviation of the EWMA statistic, in sigma units.

    ``sigma_z = sqrt(lam / (2 - lam))`` -- the limit of
    ``sqrt(lam / (2 - lam) * (1 - (1 - lam)**(2 t)))`` as ``t -> inf``
    (Montgomery 2013 ch. 9).  Dimensionless.
    """
    lam = float(lam)
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must lie in (0, 1], got {lam!r}")
    return float(np.sqrt(lam / (2.0 - lam)))


def _ewma_chain(delta: float, lam: float, limit_mult: float, n_states: int):
    """Lucas-Saccucci transition matrix for the two-sided EWMA statistic.

    The in-control region ``(-c, c)`` with ``c = limit_mult * sigma_z`` is split
    into ``n_states`` equal intervals.  From the midpoint ``t_i``,
    ``z' = (1 - lam) t_i + lam x`` with ``x ~ N(delta, 1)``, so
    ``P(z' < e) = Phi((e - (1 - lam) t_i) / lam - delta)``.
    """
    lam = float(lam)
    c = float(limit_mult) * ewma_sigma_z(lam)
    edges = np.linspace(-c, c, n_states + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    def rows(origins: NDArray[np.float64]) -> NDArray[np.float64]:
        arg = (edges[None, :] - (1.0 - lam) * origins[:, None]) / lam - float(delta)
        return np.diff(norm.cdf(arg), axis=1)

    return rows(mids), rows(np.zeros(1))[0]


def ewma_arl_markov(delta: float, lam: float, limit_mult: float, n_states: int = 401) -> float:
    """Two-sided EWMA ARL by the Lucas-Saccucci (1990) Markov-chain method.

    Parameters
    ----------
    delta
        Mean shift in sigma units (dimensionless).
    lam
        EWMA smoothing weight in (0, 1].
    limit_mult
        Control-limit multiplier ``L``; limits are ``+/- L * sigma_z`` with
        ``sigma_z`` from :func:`ewma_sigma_z` (steady-state, *not* time-varying).
    n_states
        Discretisation intervals across the in-control region.  An odd value
        places a state midpoint exactly at ``z = 0``.

    Returns
    -------
    float
        Expected samples until ``|z|`` leaves the control limits, starting from
        ``z = 0``.

    Notes
    -----
    Uses the steady-state (fixed) control limits.  The time-varying limits of
    :class:`telemetryool.charts.EwmaChart` with
    ``time_varying_limits=True`` give a different, shorter ARL0 because the
    early limits are tighter; this function does not model that case.
    """
    if n_states < 3:
        raise ValueError(f"n_states must be >= 3, got {n_states!r}")
    p_mat, start = _ewma_chain(delta, lam, limit_mult, n_states)
    mu = np.linalg.solve(np.eye(n_states) - p_mat, np.ones(n_states))
    return float(1.0 + start @ mu)


def ewma_window_false_alarm(
    lam: float, limit_mult: float, window_length: int, n_states: int = 301, delta: float = 0.0
) -> float:
    """Probability that a two-sided EWMA alarms within ``window_length`` samples.

    Exact for the discretised chain; see :func:`ewma_arl_markov` for the model.
    """
    window_length = _check_window(window_length)
    p_mat, start = _ewma_chain(delta, lam, limit_mult, n_states)
    return float(1.0 - _survival(p_mat, start, window_length))


def design_ewma_L(
    target_alpha_w: float,
    lam: float,
    window_length: int,
    n_states: int = 301,
    bracket: tuple[float, float] = (0.2, 12.0),
    tol: float = 1e-10,
) -> ChartDesign:
    """Solve for the EWMA control-limit multiplier ``L`` delivering ``target_alpha_w``."""
    target = _check_prob("target_alpha_w", target_alpha_w)
    window_length = _check_window(window_length)
    lo, hi = bracket

    def f(mult: float) -> float:
        return ewma_window_false_alarm(lam, mult, window_length, n_states) - target

    if f(lo) < 0 or f(hi) > 0:
        raise ValueError(
            f"target_alpha_w={target} is not reachable for lam={lam}, W={window_length} "
            f"within L in {bracket}: alpha_W spans "
            f"[{f(hi) + target:.3e}, {f(lo) + target:.3e}]"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    mult = 0.5 * (lo + hi)
    return ChartDesign(
        threshold=mult,
        target_alpha_w=target,
        achieved_alpha_w=ewma_window_false_alarm(lam, mult, window_length, n_states),
        window_length=window_length,
        arl0=ewma_arl_markov(0.0, lam, mult, max(n_states, 401)),
        method=f"lucas-saccucci markov chain, n_states={n_states}",
    )


# --------------------------------------------------------------------------- #
# Out-of-limit checking with persistence
# --------------------------------------------------------------------------- #


def ool_window_false_alarm(p_exceed: float, persistence: int, window_length: int) -> float:
    """Exact probability that ``persistence`` consecutive limit breaches occur in ``W``.

    The persistence counter is a Markov chain on states ``0 .. persistence - 1``
    with absorption on reaching ``persistence``: from any state the counter
    advances with probability ``p_exceed`` and resets to 0 otherwise.  This is
    exact (no discretisation) for independent samples.

    Parameters
    ----------
    p_exceed
        Per-sample probability that the value lies outside the limit,
        dimensionless, in (0, 1).  For a two-sided limit at ``+/- L sigma`` on
        standard normal data this is ``2 * (1 - Phi(L))``.
    persistence
        Consecutive breaching samples required to raise, >= 1.
    window_length
        ``W``, samples in the window, >= 1.

    Returns
    -------
    float
        ``P(at least one alarm in W samples)``, dimensionless.
    """
    p = _check_prob("p_exceed", p_exceed)
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence!r}")
    window_length = _check_window(window_length)
    n = int(persistence)
    p_mat = np.zeros((n, n))
    p_mat[:, 0] = 1.0 - p
    for i in range(n - 1):
        p_mat[i, i + 1] = p
    # Row n-1 advances into the absorbing state with probability p; that mass is
    # simply absent from p_mat, which is what makes the survival sum < 1.
    start = np.zeros(n)
    start[0] = 1.0
    v = start @ p_mat
    for _ in range(window_length - 1):
        v = v @ p_mat
    return float(1.0 - v.sum())


def ool_arl(p_exceed: float, persistence: int) -> float:
    """Expected samples until ``persistence`` consecutive breaches occur.

    Closed form for a run of ``r`` successes in Bernoulli trials with success
    probability ``p``: ``ARL = (1 - p**r) / (p**r * (1 - p))`` (the standard
    geometric-run expectation; Feller, *An Introduction to Probability Theory
    and Its Applications*, Vol. 1, 3rd ed., ch. XIII, renewal argument).
    """
    p = _check_prob("p_exceed", p_exceed)
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence!r}")
    r = int(persistence)
    return float((1.0 - p**r) / (p**r * (1.0 - p)))


def design_ool_limit(
    target_alpha_w: float,
    persistence: int,
    window_length: int,
    two_sided: bool = True,
    tol: float = 1e-14,
) -> ChartDesign:
    """Solve for the limit multiplier ``L`` delivering ``target_alpha_w``.

    Bisection on ``p_exceed`` through :func:`ool_window_false_alarm`, then
    ``L = Phi^-1(1 - p / 2)`` for a two-sided limit or ``Phi^-1(1 - p)`` for a
    one-sided one.  Assumes standard normal in-control data.
    """
    target = _check_prob("target_alpha_w", target_alpha_w)
    window_length = _check_window(window_length)
    lo, hi = 1e-12, 1.0 - 1e-12

    def f(p: float) -> float:
        return ool_window_false_alarm(p, persistence, window_length) - target

    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    p = 0.5 * (lo + hi)
    mult = float(norm.ppf(1.0 - (p / 2.0 if two_sided else p)))
    return ChartDesign(
        threshold=mult,
        target_alpha_w=target,
        achieved_alpha_w=ool_window_false_alarm(p, persistence, window_length),
        window_length=window_length,
        arl0=ool_arl(p, persistence),
        method="exact persistence-counter markov chain",
    )
