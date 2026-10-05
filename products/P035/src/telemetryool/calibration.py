"""Threshold calibration to a target window false-alarm probability, and its error bars.

The central object is the *window trigger level*.  For a detector that alarms
when its score exceeds a threshold ``h`` for ``persistence`` consecutive
samples, a monitoring window of scores ``s_0 .. s_{W-1}`` alarms if and only if

    h  <  max_t  min(s_t, s_{t+1}, ..., s_{t+persistence-1})

The right-hand side depends only on the scores, not on ``h``.  Calling it
``w`` gives three things at once:

* calibration is an empirical quantile of ``w`` over nominal windows, exact
  rather than bisected;
* the ROC is one sorted sweep over the nominal and anomalous ``w`` vectors;
* the window alarm indicator is a Bernoulli trial, so every rate reported here
  carries an exact binomial standard error.

Sample-size choice
------------------
A window false-alarm probability near ``alpha`` estimated from ``M``
independent windows has standard error ``sqrt(alpha (1 - alpha) / M)``.  At
``alpha = 0.05`` that is ``0.00154`` for ``M = 20000`` (3.1 % relative) and
``0.00689`` for ``M = 1000`` (13.8 % relative).  The package default of
``M = 20000`` is chosen so that a 10 % relative discrepancy between design and
measurement is a 3-sigma event rather than noise.  :func:`windows_for_precision`
inverts the relation.

References
----------
Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and
    Statistical Inference." *Journal of the American Statistical Association*
    22(158), 209-212.  The score interval used by :func:`wilson_interval`.
Clopper, C. J. and Pearson, E. S. (1934). "The Use of Confidence or Fiducial
    Limits Illustrated in the Case of the Binomial." *Biometrika* 26(4),
    404-413.  The exact interval used by :func:`clopper_pearson_interval`.
Brown, L. D., Cai, T. T. and DasGupta, A. (2001). "Interval Estimation for a
    Binomial Proportion." *Statistical Science* 16(2), 101-133.  Recommends the
    Wilson interval over the Wald interval at small ``n`` or extreme ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, norm

__all__ = [
    "window_trigger_level",
    "Calibration",
    "calibrate_threshold",
    "RateEstimate",
    "estimate_rate",
    "binomial_se",
    "wilson_interval",
    "clopper_pearson_interval",
    "windows_for_precision",
]


def window_trigger_level(
    scores: NDArray[np.float64], persistence: int
) -> NDArray[np.float64]:
    """Largest threshold at which each window would still raise an alarm.

    Parameters
    ----------
    scores
        Shape ``(n_windows, window_length)``; higher means more anomalous.  Any
        units, as long as they are consistent with the threshold.
    persistence
        Consecutive exceeding samples required to raise, >= 1.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_windows,)``.  Element ``i`` is
        ``max_t min(s[i, t:t+persistence])``, or ``-inf`` if the window is
        shorter than ``persistence``.  A window alarms at threshold ``h`` if and
        only if ``w[i] > h``.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_windows, window_length), got {arr.shape}")
    if not isinstance(persistence, (int, np.integer)) or persistence < 1:
        raise ValueError(f"persistence must be an integer >= 1, got {persistence!r}")
    p = int(persistence)
    n_win, length = arr.shape
    if length < p:
        return np.full(n_win, -np.inf)
    if p == 1:
        return arr.max(axis=1)
    # Rolling minimum over p consecutive samples, then the maximum over t.
    # p is small (debounce counts are single digits), so the direct fold is
    # cheaper than a monotone deque and stays fully vectorised.
    rolling = arr[:, : length - p + 1].copy()
    for j in range(1, p):
        np.minimum(rolling, arr[:, j : length - p + 1 + j], out=rolling)
    return rolling.max(axis=1)


@dataclass(frozen=True)
class Calibration:
    """A threshold calibrated against nominal data.

    Attributes
    ----------
    threshold
        Score threshold.  A window alarms when the score exceeds it for
        ``persistence`` consecutive samples.
    persistence
        Debounce count used.
    target_alpha_w
        Requested window false-alarm probability.
    achieved_alpha_w
        Fraction of the *calibration* windows that alarm at ``threshold``.  This
        is in-sample and is therefore not the measured false-alarm rate; an
        independent measurement is what
        :func:`telemetryool.harness.run_comparison` reports.
    n_calibration_windows
        Number of nominal windows used.
    n_alarming
        Number of those windows that alarm at ``threshold``.
    """

    threshold: float
    persistence: int
    target_alpha_w: float
    achieved_alpha_w: float
    n_calibration_windows: int
    n_alarming: int


def calibrate_threshold(
    nominal_scores: NDArray[np.float64],
    persistence: int,
    target_alpha_w: float,
) -> Calibration:
    """Threshold whose in-sample window alarm rate is as close as possible to the target.

    The threshold is the ``ceil(target * M)``-th largest window trigger level,
    shifted down by one representable step so that exactly that many windows
    satisfy ``w > threshold`` when there are no ties.  With ties the achieved
    rate can only be the discrete value the data permit, and
    ``achieved_alpha_w`` reports it rather than the request.

    Parameters
    ----------
    nominal_scores
        Shape ``(n_windows, window_length)`` scores on *nominal* data.
    persistence
        Debounce count, >= 1.
    target_alpha_w
        Target window false-alarm probability in (0, 1).

    Returns
    -------
    Calibration

    Raises
    ------
    ValueError
        If ``target_alpha_w * n_windows < 1``, in which case no threshold on this
        sample can represent the target and a larger calibration set is needed;
        or if ``persistence`` exceeds the window length, in which case no window
        can alarm at any threshold.
    """
    target = float(target_alpha_w)
    if not (0.0 < target < 1.0):
        raise ValueError(f"target_alpha_w must lie in (0, 1), got {target_alpha_w!r}")
    w = window_trigger_level(nominal_scores, persistence)
    m = w.size
    if not np.any(np.isfinite(w)):
        raise ValueError(
            f"no window can ever alarm: persistence={persistence} exceeds the window "
            f"length {np.shape(nominal_scores)[1]}, so no threshold delivers "
            f"alpha_W={target}"
        )
    if target * m < 1.0:
        raise ValueError(
            f"target_alpha_w={target} needs at least {int(np.ceil(1.0 / target))} "
            f"calibration windows to be representable; got {m}"
        )
    k = int(np.ceil(target * m))
    order = np.sort(w)[::-1]
    cut = float(order[k - 1])
    thresh = float(np.nextafter(cut, -np.inf))
    n_alarm = int((w > thresh).sum())
    return Calibration(
        threshold=thresh,
        persistence=int(persistence),
        target_alpha_w=target,
        achieved_alpha_w=n_alarm / m,
        n_calibration_windows=m,
        n_alarming=n_alarm,
    )


@dataclass(frozen=True)
class RateEstimate:
    """A binomial rate with its standard error and two interval estimates.

    Attributes
    ----------
    successes, trials
        Counts.
    rate
        ``successes / trials``, dimensionless.
    standard_error
        ``sqrt(rate (1 - rate) / trials)``, dimensionless.
    wilson_low, wilson_high
        Wilson score interval at ``confidence``.
    exact_low, exact_high
        Clopper-Pearson interval at ``confidence``.
    confidence
        Nominal coverage, e.g. 0.95.
    """

    successes: int
    trials: int
    rate: float
    standard_error: float
    wilson_low: float
    wilson_high: float
    exact_low: float
    exact_high: float
    confidence: float

    def z_against(self, design: float) -> float:
        """Signed discrepancy from ``design`` in units of the standard error.

        Uses the standard error under the *design* value, ``sqrt(p0(1-p0)/n)``,
        which is the correct null standard error for testing agreement.
        Returns ``nan`` if the design value is 0 or 1.
        """
        p0 = float(design)
        if not (0.0 < p0 < 1.0):
            return float("nan")
        se0 = np.sqrt(p0 * (1.0 - p0) / self.trials)
        return float((self.rate - p0) / se0)


def binomial_se(rate: float, trials: int) -> float:
    """``sqrt(rate (1 - rate) / trials)``, dimensionless."""
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    r = float(rate)
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"rate must lie in [0, 1], got {rate!r}")
    return float(np.sqrt(r * (1.0 - r) / trials))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion."""
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    if not (0 <= successes <= trials):
        raise ValueError(f"successes must lie in [0, {trials}], got {successes!r}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    z = float(norm.ppf(0.5 * (1.0 + confidence)))
    n = float(trials)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Clopper-Pearson (1934) exact interval for a binomial proportion."""
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    if not (0 <= successes <= trials):
        raise ValueError(f"successes must lie in [0, {trials}], got {successes!r}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    a = 1.0 - confidence
    lo = 0.0 if successes == 0 else float(beta.ppf(a / 2.0, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(
        beta.ppf(1.0 - a / 2.0, successes + 1, trials - successes)
    )
    return (lo, hi)


def estimate_rate(successes: int, trials: int, confidence: float = 0.95) -> RateEstimate:
    """Package a binomial count as a :class:`RateEstimate`."""
    rate = successes / trials
    wl, wh = wilson_interval(successes, trials, confidence)
    el, eh = clopper_pearson_interval(successes, trials, confidence)
    return RateEstimate(
        successes=int(successes),
        trials=int(trials),
        rate=float(rate),
        standard_error=binomial_se(rate, trials),
        wilson_low=wl,
        wilson_high=wh,
        exact_low=el,
        exact_high=eh,
        confidence=float(confidence),
    )


def windows_for_precision(alpha: float, relative_se: float) -> int:
    """Windows needed so that the binomial relative standard error is ``relative_se``.

    ``M = (1 - alpha) / (alpha * relative_se**2)``, rounded up.

    Examples
    --------
    >>> windows_for_precision(0.05, 0.10)
    1900
    >>> windows_for_precision(0.05, 0.031)
    19772
    """
    a = float(alpha)
    r = float(relative_se)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if r <= 0.0:
        raise ValueError(f"relative_se must be > 0, got {relative_se!r}")
    return int(np.ceil((1.0 - a) / (a * r * r)))
