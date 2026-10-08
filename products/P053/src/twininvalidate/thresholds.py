"""Thresholds set from a declared false-alarm target, never from a delay.

The rule this package follows, stated once and obeyed everywhere:

    **A threshold is chosen so that the in-control average run length (ARL0)
    of the detector hits a declared target, using in-control residual streams
    only. No out-of-control stream, and no detection delay, is visible to the
    threshold-setting procedure.**

That is the only way an ARL0-against-ARL1 curve means anything: if the
threshold were adjusted after the delay were known, the curve would be a
record of the tuning rather than of the detector.

Two methods are provided.

Closed form
-----------
Two special cases have an exact threshold, because the statistic reduces to a
per-sample test of a single standard normal:

* GLR with ``window = 1``: ``g_k = z_k^2 / 2``, so an alarm occurs iff
  ``|z_k| > sqrt(2 h)``. The per-sample false-alarm probability is
  ``p = 2 (1 - Phi(sqrt(2 h)))``, the run length is Geometric(p) with mean
  ``1/p``, and therefore

      h*(T) = Phi^{-1}(1 - 1/(2 T))^2 / 2.                              (7)

* EWMA with ``lam = 1``: ``g_k = |z_k|``, so

      L*(T) = Phi^{-1}(1 - 1/(2 T)).                                    (8)

These are used as known answers in the tests and as an independent check on
the Monte-Carlo calibrator in ``validation/validate_thresholds.py``. No
closed form is claimed for CUSUM, for EWMA with ``lam < 1`` or for GLR with
``window > 1``; published approximations exist but none was verified in this
environment, so none is quoted.

Monte Carlo
-----------
For every other case the threshold is found by bisection on the censored
maximum-likelihood ARL0 estimator of :mod:`twininvalidate.arl`, over a fixed
bank of in-control streams generated from a declared seed. The estimator is
monotone in the threshold up to Monte-Carlo noise, so bisection is well posed;
the achieved ARL0 and its standard error are returned alongside the threshold
so a reader can see how close the calibration got rather than taking it on
trust.

Units
-----
``target_arl0`` is in **samples**. :func:`arl0_from_rate` converts a false
alarms per 1000 hours figure into samples at a declared sample rate, because
that is the unit a reliability requirement is usually written in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .arl import arl0_estimate
from .detectors import DetectorSpec


def arl0_from_rate(false_alarms_per_1000h: float, sample_rate_hz: float) -> float:
    """Convert a false-alarm rate requirement into an ARL0 in samples.

    ``ARL0 = 1000 * 3600 * f / N`` for ``N`` false alarms per 1000 hours at a
    sample rate of ``f`` Hz, because the mean time between false alarms is
    ``1000 h / N`` and one sample lasts ``1/f`` s.

    Parameters
    ----------
    false_alarms_per_1000h:
        Allowed false alarms per 1000 operating hours, strictly positive.
    sample_rate_hz:
        Monitor sample rate in Hz, strictly positive.

    Returns
    -------
    Target ARL0 in samples.
    """
    if false_alarms_per_1000h <= 0.0:
        raise ValueError(
            f"false_alarms_per_1000h must be positive, got {false_alarms_per_1000h}"
        )
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    return 1000.0 * 3600.0 * sample_rate_hz / false_alarms_per_1000h


def rate_from_arl0(arl0_samples: float, sample_rate_hz: float) -> float:
    """Inverse of :func:`arl0_from_rate`: false alarms per 1000 hours."""
    if arl0_samples <= 0.0:
        raise ValueError(f"arl0_samples must be positive, got {arl0_samples}")
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    return 1000.0 * 3600.0 * sample_rate_hz / arl0_samples


def glr_window1_threshold(target_arl0: float) -> float:
    """Exact GLR(window=1) threshold for a target ARL0, equation (7).

    Parameters
    ----------
    target_arl0:
        Target in-control average run length in samples, strictly greater
        than 1.

    Returns
    -------
    Threshold in nats.
    """
    if target_arl0 <= 1.0:
        raise ValueError(f"target_arl0 must exceed 1 sample, got {target_arl0}")
    return float(norm.ppf(1.0 - 1.0 / (2.0 * target_arl0)) ** 2 / 2.0)


def ewma_lambda1_threshold(target_arl0: float) -> float:
    """Exact EWMA(lambda=1) threshold for a target ARL0, equation (8)."""
    if target_arl0 <= 1.0:
        raise ValueError(f"target_arl0 must exceed 1 sample, got {target_arl0}")
    return float(norm.ppf(1.0 - 1.0 / (2.0 * target_arl0)))


def closed_form_threshold(spec: DetectorSpec, target_arl0: float) -> float | None:
    """Closed-form threshold if one exists for this detector, else ``None``."""
    if spec.name == "glr" and spec.window == 1:
        return glr_window1_threshold(target_arl0)
    if spec.name == "ewma" and spec.lam == 1.0:
        return ewma_lambda1_threshold(target_arl0)
    return None


@dataclass(frozen=True)
class Calibration:
    """The outcome of setting one threshold.

    Attributes
    ----------
    spec:
        The detector the threshold belongs to.
    threshold:
        The chosen threshold, in the statistic's units.
    target_arl0:
        The declared target, samples.
    achieved_arl0:
        Censored-MLE ARL0 of the chosen threshold on the calibration streams,
        samples.
    achieved_stderr:
        Standard error of ``achieved_arl0``, samples.
    n_alarms:
        Number of calibration runs that alarmed. A calibration resting on a
        handful of alarms is not worth much and this field is what says so.
    n_runs, n_samples:
        Size of the calibration bank.
    method:
        ``"closed-form"`` or ``"monte-carlo-bisection"``.
    """

    spec: DetectorSpec
    threshold: float
    target_arl0: float
    achieved_arl0: float
    achieved_stderr: float
    n_alarms: int
    n_runs: int
    n_samples: int
    method: str

    @property
    def relative_error(self) -> float:
        """``achieved/target - 1``, dimensionless."""
        return self.achieved_arl0 / self.target_arl0 - 1.0


def calibrate_threshold(
    spec: DetectorSpec,
    in_control: np.ndarray,
    target_arl0: float,
    *,
    bracket: tuple[float, float] | None = None,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> Calibration:
    """Set ``spec``'s threshold so its ARL0 on ``in_control`` hits the target.

    Parameters
    ----------
    spec:
        Detector and its declared design constants.
    in_control:
        In-control normalised residual streams, shape ``(n_runs, n_samples)``.
        These must come from a stream whose change kind is ``none``; nothing
        here checks that, so the caller carries the responsibility.
    target_arl0:
        Target in-control ARL0 in samples. Must exceed 1 and should be well
        below ``n_runs * n_samples`` or the calibration will rest on too few
        alarms to mean anything.
    bracket:
        Optional ``(low, high)`` threshold bracket. Defaults to
        ``(0, 1.05 * max(statistic))``.
    tol:
        Absolute bisection tolerance on the threshold.
    max_iter:
        Maximum bisection iterations.

    Returns
    -------
    A :class:`Calibration`.

    Notes
    -----
    The ARL0 estimate used inside the bisection is the censored MLE, so runs
    that never alarm contribute their full horizon to the time at risk instead
    of being discarded.
    """
    if target_arl0 <= 1.0:
        raise ValueError(f"target_arl0 must exceed 1 sample, got {target_arl0}")
    stat = spec.statistic(in_control)
    n_runs, n_samples = stat.shape

    closed = closed_form_threshold(spec, target_arl0)
    if closed is not None:
        est = arl0_estimate(stat, closed)
        return Calibration(
            spec=spec,
            threshold=closed,
            target_arl0=float(target_arl0),
            achieved_arl0=est.value,
            achieved_stderr=est.stderr,
            n_alarms=est.n_detected,
            n_runs=n_runs,
            n_samples=n_samples,
            method="closed-form",
        )

    low, high = bracket if bracket is not None else (0.0, 1.05 * float(stat.max()))
    if high <= low:
        raise ValueError(f"bracket must be increasing, got {(low, high)}")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        est = arl0_estimate(stat, mid)
        if est.value < target_arl0:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    threshold = 0.5 * (low + high)
    est = arl0_estimate(stat, threshold)
    return Calibration(
        spec=spec,
        threshold=float(threshold),
        target_arl0=float(target_arl0),
        achieved_arl0=est.value,
        achieved_stderr=est.stderr,
        n_alarms=est.n_detected,
        n_runs=n_runs,
        n_samples=n_samples,
        method="monte-carlo-bisection",
    )


@dataclass(frozen=True)
class ThresholdBracket:
    """The two achievable thresholds that bracket a target ARL0.

    A statistic with finitely many distinct values cannot be set to an
    arbitrary false-alarm rate. An isotonic-calibrated classifier is exactly
    that case: its confidence takes a few thousand distinct values, so between
    one achievable in-control ARL0 and the next there may be a factor of 1.3 or
    more, and a bisection that stops at "the highest threshold whose ARL0 is
    still below the target" quietly gives that detector more false alarms than
    its competitors.

    This object makes the gap visible. ``conservative`` is the smallest
    achievable threshold whose ARL0 is at least the target -- the one that is
    comparable with a continuously-adjustable detector -- and ``generous`` is
    the largest whose ARL0 is below it. Both are reported in
    ``validation/validate_classifier.py``; the conservative one is the primary
    comparison.

    Attributes
    ----------
    conservative, generous:
        Thresholds in the statistic's units. ``generous`` is ``None`` if every
        achievable threshold already meets the target.
    conservative_arl0, generous_arl0:
        Achieved censored-MLE ARL0 in samples.
    n_levels:
        Number of distinct values the statistic took on the calibration bank.
    """

    conservative: float
    conservative_arl0: float
    generous: float | None
    generous_arl0: float | None
    n_levels: int


def bracket_threshold(
    statistic: np.ndarray, target_arl0: float, max_levels: int = 400
) -> ThresholdBracket:
    """Find the achievable thresholds bracketing ``target_arl0``.

    Candidate thresholds are taken from the distinct values the statistic
    actually attains, because a threshold strictly between two attained values
    gives the same alarms as the lower of them. For statistics with many
    distinct values the candidate list is thinned to ``max_levels`` quantiles
    of the upper tail, which is enough to resolve the bracket and keeps the
    search affordable.

    Parameters
    ----------
    statistic:
        In-control statistic paths, shape ``(n_runs, n_samples)``.
    target_arl0:
        Target in-control ARL0 in samples.
    max_levels:
        Maximum number of candidate thresholds to evaluate.
    """
    if target_arl0 <= 1.0:
        raise ValueError(f"target_arl0 must exceed 1 sample, got {target_arl0}")
    stat = np.asarray(statistic, dtype=float)
    levels = np.unique(stat)
    n_levels = int(levels.size)
    if n_levels > max_levels:
        qs = np.linspace(50.0, 100.0, max_levels)
        levels = np.unique(np.percentile(stat, qs))
    arls = np.array([arl0_estimate(stat, float(t)).value for t in levels])
    # An infinite estimate means no run alarmed at that threshold, which is an
    # absence of evidence, not evidence that the target is met. Such levels are
    # excluded, so a bank too small to resolve the target raises instead of
    # silently returning the largest threshold it happened to try.
    finite = np.isfinite(arls)
    meets = finite & (arls >= target_arl0)
    if not meets.any():
        largest = float(np.max(arls[finite])) if finite.any() else float("nan")
        raise ValueError(
            f"no achievable threshold reaches ARL0 {target_arl0:g} with at "
            f"least one alarm on this calibration bank; the largest resolvable "
            f"ARL0 is {largest:.1f}. Simulate more in-control samples."
        )
    i = int(np.argmax(meets))
    conservative = float(levels[i])
    conservative_arl0 = float(arls[i])
    below = np.nonzero(finite & (arls < target_arl0) & (np.arange(levels.size) < i))[0]
    if below.size == 0:
        return ThresholdBracket(conservative, conservative_arl0, None, None, n_levels)
    j = int(below[-1])
    return ThresholdBracket(
        conservative, conservative_arl0, float(levels[j]), float(arls[j]), n_levels
    )
