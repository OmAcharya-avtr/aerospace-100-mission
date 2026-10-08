"""Average run lengths and the detection-delay against false-alarm curve.

Definitions used throughout, all in **samples**:

ARL0
    In-control average run length: the expected number of samples from the
    start of monitoring to the first alarm when the twin still describes the
    asset. Large is good. ``1 / ARL0`` is the per-sample false-alarm
    probability, and :func:`twininvalidate.thresholds.rate_from_arl0` converts
    it to false alarms per 1000 hours.
ARL1
    Out-of-control average run length, here the **zero-state** detection
    delay: the change is present from sample 0 and the detector statistic
    starts at zero, so the alarm index is the delay. Small is good. This is
    the standard zero-state convention (Basseville & Nikiforov 1993); the
    steady-state convention, in which the detector has already been running
    when the change arrives, gives a slightly smaller delay and is not what is
    reported here.

Censoring
---------
A simulation runs for a finite horizon, so some runs never alarm. Dropping
them biases every average downwards, which would make every detector look
better than it is. Both estimators below handle it explicitly:

* :func:`arl0_estimate` uses the right-censored maximum-likelihood estimator
  for a geometric run length,

      ARL0_hat = (total samples at risk) / (number of alarms),            (9)

  where a censored run contributes its full horizon to the numerator and
  nothing to the denominator. For a detector whose alarms really are a
  Bernoulli process -- which GLR with ``window = 1`` exactly is -- (9) is the
  MLE of ``1/p``, and ``validation/validate_thresholds.py`` checks it against
  the closed form.
* :func:`arl1_estimate` reports the mean delay over detected runs **and** the
  detection fraction, and additionally a lower bound in which censored runs
  contribute the full horizon. A delay quoted without its detection fraction
  is not interpretable and this package does not quote one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import DetectorSpec, first_alarm


@dataclass(frozen=True)
class ArlEstimate:
    """One average-run-length estimate with its uncertainty and censoring.

    Attributes
    ----------
    value:
        The estimate, in samples.
    stderr:
        Standard error of ``value``, in samples.
    n_runs:
        Runs simulated.
    n_detected:
        Runs that alarmed within the horizon.
    horizon:
        Samples simulated per run.
    lower_bound:
        For ARL1, the mean delay with censored runs counted at ``horizon``;
        equal to ``value`` when nothing was censored. For ARL0 this is the
        plain censored mean, which under-estimates ARL0.
    """

    value: float
    stderr: float
    n_runs: int
    n_detected: int
    horizon: int
    lower_bound: float

    @property
    def detection_fraction(self) -> float:
        """Fraction of runs that alarmed within the horizon."""
        return self.n_detected / self.n_runs if self.n_runs else float("nan")

    @property
    def censored_fraction(self) -> float:
        """Fraction of runs that did not alarm within the horizon."""
        return 1.0 - self.detection_fraction


def arl0_estimate(statistic: np.ndarray, threshold: float) -> ArlEstimate:
    """Censored-MLE in-control ARL0, equation (9), in samples.

    Parameters
    ----------
    statistic:
        In-control statistic paths, shape ``(n_runs, n_samples)``.
    threshold:
        Alarm threshold.

    Returns
    -------
    An :class:`ArlEstimate`. If no run alarmed, ``value`` is ``inf`` and
    ``stderr`` is ``nan``: the data cannot bound the ARL0 from above, and
    returning the horizon instead would be a fabrication.
    """
    idx = first_alarm(statistic, threshold)
    horizon = int(np.asarray(statistic).reshape(idx.size, -1).shape[1])
    detected = idx >= 0
    n_detected = int(detected.sum())
    # Samples at risk: an alarm at index i means i+1 samples were observed.
    at_risk = float(np.where(detected, idx + 1, horizon).sum())
    if n_detected == 0:
        return ArlEstimate(
            value=float("inf"),
            stderr=float("nan"),
            n_runs=int(idx.size),
            n_detected=0,
            horizon=horizon,
            lower_bound=float(horizon),
        )
    value = at_risk / n_detected
    # Geometric run length: sd(run length) = sqrt(1-p)/p <= 1/p = ARL0, and the
    # MLE of 1/p from n_detected events has relative standard error
    # 1/sqrt(n_detected).
    stderr = value / np.sqrt(n_detected)
    censored_mean = float(np.where(detected, idx + 1, horizon).mean())
    return ArlEstimate(
        value=float(value),
        stderr=float(stderr),
        n_runs=int(idx.size),
        n_detected=n_detected,
        horizon=horizon,
        lower_bound=censored_mean,
    )


def arl1_estimate(statistic: np.ndarray, threshold: float) -> ArlEstimate:
    """Zero-state detection delay in samples, with its detection fraction.

    ``value`` is the mean alarm index plus one over the runs that alarmed;
    ``lower_bound`` is the same mean with censored runs counted at the horizon,
    which is a lower bound on the true mean delay.

    Parameters
    ----------
    statistic:
        Out-of-control statistic paths with the change present from sample 0.
    threshold:
        Alarm threshold.
    """
    idx = first_alarm(statistic, threshold)
    horizon = int(np.asarray(statistic).reshape(idx.size, -1).shape[1])
    detected = idx >= 0
    n_detected = int(detected.sum())
    bounded = np.where(detected, idx + 1, horizon).astype(float)
    if n_detected == 0:
        return ArlEstimate(
            value=float("inf"),
            stderr=float("nan"),
            n_runs=int(idx.size),
            n_detected=0,
            horizon=horizon,
            lower_bound=float(horizon),
        )
    delays = (idx[detected] + 1).astype(float)
    value = float(delays.mean())
    stderr = float(delays.std(ddof=1) / np.sqrt(n_detected)) if n_detected > 1 else float("nan")
    return ArlEstimate(
        value=value,
        stderr=stderr,
        n_runs=int(idx.size),
        n_detected=n_detected,
        horizon=horizon,
        lower_bound=float(bounded.mean()),
    )


@dataclass(frozen=True)
class CurvePoint:
    """One point of a detection-delay against false-alarm-rate curve."""

    threshold: float
    arl0: ArlEstimate
    arl1: ArlEstimate

    @property
    def row(self) -> tuple[float, float, float, float]:
        """``(threshold, arl0, arl1, detection_fraction)``, for tables."""
        return (self.threshold, self.arl0.value, self.arl1.value, self.arl1.detection_fraction)


def delay_curve(
    spec: DetectorSpec,
    in_control: np.ndarray,
    out_of_control: np.ndarray,
    thresholds: np.ndarray,
) -> list[CurvePoint]:
    """Sweep thresholds over stored statistic paths and return the curve.

    The two statistic paths are computed once and every threshold is evaluated
    against the same stored matrices, which is why a full curve costs about the
    same as a single point.

    Parameters
    ----------
    spec:
        Detector and declared design constants.
    in_control:
        In-control residual streams, shape ``(n_runs, n_samples)``.
    out_of_control:
        Residual streams with the change present from sample 0.
    thresholds:
        Thresholds to evaluate, in the statistic's units.

    Returns
    -------
    One :class:`CurvePoint` per threshold, in the order given.
    """
    thr = np.atleast_1d(np.asarray(thresholds, dtype=float))
    if thr.size == 0:
        raise ValueError("thresholds must contain at least one value")
    if np.any(thr < 0.0):
        raise ValueError("thresholds must be non-negative for these statistics")
    stat_ic = spec.statistic(in_control)
    stat_oc = spec.statistic(out_of_control)
    return [
        CurvePoint(
            threshold=float(t),
            arl0=arl0_estimate(stat_ic, float(t)),
            arl1=arl1_estimate(stat_oc, float(t)),
        )
        for t in thr
    ]


def threshold_grid(statistic: np.ndarray, n_points: int = 14) -> np.ndarray:
    """A geometric threshold grid spanning the useful range of a statistic.

    Spans the 50th to the 99.99th percentile of the statistic values, which
    brackets the in-control ARL0 range that a finite simulation can resolve.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    stat = np.asarray(statistic, dtype=float)
    lo = float(np.percentile(stat, 50.0))
    hi = float(np.percentile(stat, 99.99))
    lo = max(lo, 1e-6 * max(hi, 1.0))
    if hi <= lo:
        raise ValueError("statistic has no spread; cannot build a threshold grid")
    return np.geomspace(lo, hi, n_points)


def delay_after_onset(
    statistic: np.ndarray, threshold: float, onset: int
) -> tuple[ArlEstimate, int]:
    """Detection delay measured from a change at sample ``onset``.

    This is the **steady-state** protocol used for the learned-model
    comparison. Every detector is given ``onset`` samples of in-control data
    first, so the windowed classifier has a full feature window at the change
    point and the recursive baselines have a warmed-up statistic. Runs that
    alarm before ``onset`` are pre-change false alarms: they are excluded from
    the delay and counted separately, because a false alarm is not a detection
    and averaging it in would flatter the detector.

    Parameters
    ----------
    statistic:
        Statistic paths, shape ``(n_runs, n_samples)``, with the change
        beginning at sample ``onset``.
    threshold:
        Alarm threshold.
    onset:
        Sample index at which the change begins, ``0 <= onset < n_samples``.

    Returns
    -------
    ``(estimate, n_pre_onset_alarms)``. The estimate's delay is measured in
    samples with 1 meaning "alarmed on the first post-onset sample", and its
    ``n_runs`` counts only the runs that survived to ``onset``.
    """
    stat = np.asarray(statistic, dtype=float)
    if stat.ndim != 2:
        raise ValueError(f"statistic must be 2-D, got ndim={stat.ndim}")
    if not 0 <= onset < stat.shape[1]:
        raise ValueError(f"onset must lie in [0, {stat.shape[1]}), got {onset}")
    pre = first_alarm(stat[:, :onset], threshold) if onset > 0 else np.full(stat.shape[0], -1)
    survived = pre < 0
    n_pre = int((~survived).sum())
    if not survived.any():
        raise ValueError(
            "every run raised a false alarm before the change onset; the "
            "threshold is far too low for this protocol"
        )
    return arl1_estimate(stat[survived, onset:], threshold), n_pre
