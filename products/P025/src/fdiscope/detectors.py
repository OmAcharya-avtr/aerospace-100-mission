"""Classical residual-based detectors: chi-squared and CUSUM.

Both operate on the *normalised* residual sequence ``r_k = L^-1 y_k`` produced
by :func:`fdiscope.simulate.simulate_loop`, which is ``N(0, I_m)`` under the
fault-free hypothesis.

The chi-squared test is the snapshot test: it sums ``|r_k|^2`` over a sliding
window and compares with a threshold read straight off the chi-squared
distribution.  It is memoryless beyond the window and its false-alarm rate per
window is a design parameter, not a tuning knob.

The CUSUM is the sequential test: it accumulates log-likelihood ratio evidence
for a *specific* mean shift along a *specific* residual direction, so it
detects small persistent faults a windowed test cannot see, at the cost of
needing that direction and shift size in advance.

References
----------
Page, E. S., "Continuous Inspection Schemes", *Biometrika*, 41(1/2), 1954,
    pp. 100-115.
Basseville, M. and Nikiforov, I. V., *Detection of Abrupt Changes: Theory and
    Application*, Prentice-Hall, 1993.
Willsky, A. S., "A Survey of Design Methods for Failure Detection in Dynamic
    Systems", *Automatica*, 12(6), 1976, pp. 601-611.
Mehra, R. K. and Peschon, J., "An Innovations Approach to Fault Detection and
    Diagnosis in Dynamic Systems", *Automatica*, 7(5), 1971, pp. 637-640.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .analytic import chi2_threshold

__all__ = [
    "DetectorOutput",
    "ChiSquaredDetector",
    "CusumDetector",
    "CusumBank",
    "first_alarm_index",
    "detection_delay",
]


@dataclass(frozen=True)
class DetectorOutput:
    """Per-sample detector output.

    Attributes
    ----------
    statistic : ndarray, shape (N,)
        The test statistic at each sample, dimensionless.  Samples before the
        detector has enough history hold ``numpy.nan``.
    alarm : ndarray of bool, shape (N,)
        ``statistic > threshold``.  ``nan`` statistics never alarm.
    threshold : float
        The threshold in use, dimensionless.
    label : str
        Human-readable detector name.
    """

    statistic: NDArray[np.float64]
    alarm: NDArray[np.bool_]
    threshold: float
    label: str = ""

    @property
    def alarm_fraction(self) -> float:
        """Fraction of *valid* samples that alarm, dimensionless."""
        valid = np.isfinite(self.statistic)
        if not np.any(valid):
            return float("nan")
        return float(np.mean(self.alarm[valid]))


def _residual_array(residual: ArrayLike) -> NDArray[np.float64]:
    r = np.atleast_2d(np.asarray(residual, dtype=float))
    if r.ndim != 2:
        raise ValueError(f"residual must be 2-D (N, m), got shape {r.shape}")
    if r.shape[0] < 1:
        raise ValueError("residual must have at least one sample")
    if not np.all(np.isfinite(r)):
        raise ValueError("residual must be finite")
    return r


@dataclass(frozen=True)
class ChiSquaredDetector:
    """Sliding-window chi-squared test on the normalised innovation squared.

    The statistic is ``T_k = sum_{i=k-W+1}^{k} |r_i|^2``, which is
    chi-squared with ``W * m`` degrees of freedom under ``H0``.  The threshold
    is ``chi2.isf(alpha, W * m)``.

    Parameters
    ----------
    window : int
        Window length ``W`` in samples.  Must be >= 1.
    dim : int
        Measurement dimension ``m``.  Must be >= 1.
    alpha : float
        Design false-alarm probability per window, in ``(0, 1)``.

    Notes
    -----
    ``alpha`` is the probability that **one** window exceeds the threshold.
    Overlapping windows are strongly correlated, so the rate of alarm *events*
    over a long fault-free run is not ``alpha``; both numbers are measured in
    ``validation/chi2_false_alarm.py`` and both are reported.
    """

    window: int = 20
    dim: int = 2
    alpha: float = 1.0e-3

    def __post_init__(self) -> None:
        if int(self.window) < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if int(self.dim) < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")

    @property
    def dof(self) -> int:
        """Degrees of freedom ``W * m``."""
        return int(self.window) * int(self.dim)

    @property
    def threshold(self) -> float:
        """Design threshold ``chi2.isf(alpha, dof)``, dimensionless."""
        return chi2_threshold(self.alpha, self.dof)

    def run(self, residual: ArrayLike) -> DetectorOutput:
        """Apply the test to a normalised residual sequence.

        Parameters
        ----------
        residual : array_like, shape (N, m)
            Normalised residuals.  ``m`` must equal ``self.dim``.

        Returns
        -------
        DetectorOutput
            The first ``W - 1`` statistics are ``nan``.
        """
        r = _residual_array(residual)
        if r.shape[1] != int(self.dim):
            raise ValueError(f"residual has {r.shape[1]} columns, expected {self.dim}")
        nis = np.sum(r * r, axis=1)
        w = int(self.window)
        stat = np.full(r.shape[0], np.nan)
        if r.shape[0] >= w:
            csum = np.concatenate(([0.0], np.cumsum(nis)))
            stat[w - 1 :] = csum[w:] - csum[:-w]
        h = self.threshold
        alarm = np.zeros(r.shape[0], dtype=bool)
        valid = np.isfinite(stat)
        alarm[valid] = stat[valid] > h
        return DetectorOutput(statistic=stat, alarm=alarm, threshold=h, label="chi2")


@dataclass(frozen=True)
class CusumDetector:
    """One-sided CUSUM on the residual projected onto a fixed direction.

    The projected sequence ``p_k = phi . r_k`` is ``N(0, 1)`` under ``H0`` for
    a unit ``phi``, and ``N(mu, 1)`` when the residual mean is ``mu phi``.
    The log-likelihood-ratio increment is ``s_k = mu p_k - mu^2 / 2`` and the
    statistic is ``g_k = max(0, g_{k-1} + s_k)`` (Page 1954).

    Parameters
    ----------
    direction : array_like, shape (m,)
        Residual direction to watch.  Normalised internally; must be non-zero.
    mu : float
        Design mean shift along ``direction``, dimensionless, positive.
        Underestimating ``mu`` costs delay; overestimating it costs
        sensitivity to small faults.
    threshold : float
        Alarm threshold ``h``, dimensionless.  Use
        :func:`fdiscope.analytic.cusum_threshold_for_arl0` to pick it from a
        target mean time between false alarms.
    label : str
        Name carried into :class:`DetectorOutput`.
    """

    direction: NDArray[np.float64]
    mu: float
    threshold: float
    label: str = "cusum"

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(d))
        if d.size < 1 or not np.all(np.isfinite(d)):
            raise ValueError("direction must be a finite non-empty vector")
        if norm < 1e-12:
            raise ValueError("direction must be non-zero")
        object.__setattr__(self, "direction", d / norm)
        if not np.isfinite(self.mu) or float(self.mu) <= 0.0:
            raise ValueError(f"mu must be positive and finite, got {self.mu}")
        if not np.isfinite(self.threshold) or float(self.threshold) <= 0.0:
            raise ValueError(f"threshold must be positive and finite, got {self.threshold}")

    def project(self, residual: ArrayLike) -> NDArray[np.float64]:
        """Project residuals onto the watched direction, shape ``(N,)``."""
        r = _residual_array(residual)
        if r.shape[1] != self.direction.size:
            raise ValueError(
                f"residual has {r.shape[1]} columns, direction has {self.direction.size}"
            )
        return r @ self.direction

    def increments(self, residual: ArrayLike) -> NDArray[np.float64]:
        """Log-likelihood-ratio increments ``s_k = mu p_k - mu^2 / 2``."""
        mu = float(self.mu)
        return mu * self.project(residual) - 0.5 * mu * mu

    def run(self, residual: ArrayLike, reset_on_alarm: bool = False) -> DetectorOutput:
        """Apply the CUSUM to a normalised residual sequence.

        Parameters
        ----------
        residual : array_like, shape (N, m)
            Normalised residuals.
        reset_on_alarm : bool
            If True the statistic returns to zero on the sample after each
            alarm.  This is what makes the *rate* of alarms comparable with
            the design ``1 / ARL0``; without it a single crossing keeps the
            statistic above the threshold for many samples and the alarm
            fraction overstates the false-alarm rate by an order of magnitude.

        Notes
        -----
        Without reset the recursion has the closed form
        ``g_k = S_k - min(0, S_0, ..., S_k)`` with ``S`` the cumulative sum of
        the increments, which is what the vectorised path computes.  The reset
        variant has no such form and runs an explicit loop.
        """
        incr = self.increments(residual)
        h = float(self.threshold)
        if not reset_on_alarm:
            s = np.cumsum(incr)
            g = s - np.minimum(0.0, np.minimum.accumulate(s))
        else:
            g = np.empty(incr.size)
            acc = 0.0
            for i in range(incr.size):
                acc = max(0.0, acc + incr[i])
                g[i] = acc
                if acc > h:
                    acc = 0.0
        return DetectorOutput(statistic=g, alarm=g > h, threshold=h, label=self.label)


@dataclass(frozen=True)
class CusumBank:
    """A bank of CUSUMs, one per fault hypothesis.

    This is the classical isolation mechanism of Willsky (1976): run a matched
    sequential test for every fault you can name, alarm when any of them
    fires, and isolate to whichever has the largest statistic.

    Parameters
    ----------
    detectors : dict of str to CusumDetector
        Named hypotheses.  Insertion order fixes the column order of
        :attr:`statistics`.
    """

    detectors: dict[str, CusumDetector] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("CusumBank needs at least one detector")
        for name, det in self.detectors.items():
            if not isinstance(det, CusumDetector):
                raise TypeError(f"detector {name!r} is not a CusumDetector")

    @property
    def names(self) -> tuple[str, ...]:
        """Hypothesis names in column order."""
        return tuple(self.detectors)

    def statistics(self, residual: ArrayLike) -> NDArray[np.float64]:
        """All bank statistics, shape ``(N, n_hypotheses)``."""
        cols = [det.run(residual).statistic for det in self.detectors.values()]
        return np.stack(cols, axis=1)

    def run_lengths(self, residual: ArrayLike) -> NDArray[np.intp]:
        """Samples between successive alarms, with the whole bank reset.

        Every member is reset to zero as soon as *any* member alarms, which is
        the standard way to measure a bank's average run length.  The final
        incomplete run is dropped, so an empty result means the bank never
        alarmed and the run length is only known to exceed ``N``.
        """
        incr = np.stack(
            [det.increments(residual) for det in self.detectors.values()], axis=1
        )
        h = float(max(d.threshold for d in self.detectors.values()))
        acc = np.zeros(incr.shape[1])
        lengths: list[int] = []
        since = 0
        for k in range(incr.shape[0]):
            acc = np.maximum(0.0, acc + incr[k])
            since += 1
            if float(np.max(acc)) > h:
                lengths.append(since)
                since = 0
                acc[:] = 0.0
        return np.asarray(lengths, dtype=np.intp)

    def run(self, residual: ArrayLike) -> DetectorOutput:
        """Max-over-bank detector output.

        The threshold is the maximum of the member thresholds, so an alarm
        means at least one member has crossed its own threshold only when all
        thresholds are equal; when they differ this is the conservative
        choice and the package uses a common threshold throughout.
        """
        stats = self.statistics(residual)
        g = np.max(stats, axis=1)
        h = float(max(d.threshold for d in self.detectors.values()))
        return DetectorOutput(statistic=g, alarm=g > h, threshold=h, label="cusum_bank")

    def isolate(self, residual: ArrayLike) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Per-sample winning hypothesis index and its statistic."""
        stats = self.statistics(residual)
        idx = np.argmax(stats, axis=1)
        return idx, stats[np.arange(stats.shape[0]), idx]


def first_alarm_index(alarm: ArrayLike, start: int = 0, persistence: int = 1) -> int:
    """Index of the first alarm at or after ``start``.

    Parameters
    ----------
    alarm : array_like of bool, shape (N,)
        Per-sample alarm flags.
    start : int
        First index considered.
    persistence : int
        Number of *consecutive* alarming samples required.  ``1`` means a
        single sample is enough; larger values trade delay for robustness.

    Returns
    -------
    int
        Index of the first sample of the qualifying run, or ``-1`` if there
        is none.
    """
    a = np.asarray(alarm, dtype=bool).reshape(-1)
    p = int(persistence)
    if p < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")
    s = max(0, int(start))
    run = 0
    for i in range(s, a.size):
        run = run + 1 if a[i] else 0
        if run >= p:
            return int(i - p + 1)
    return -1


def detection_delay(
    alarm: ArrayLike, onset_step: int, persistence: int = 1
) -> float:
    """Detection delay in samples, or ``inf`` if the fault is never detected.

    Parameters
    ----------
    alarm : array_like of bool
        Per-sample alarm flags.
    onset_step : int
        Index of the first faulted sample.
    persistence : int
        As in :func:`first_alarm_index`.

    Returns
    -------
    float
        ``first_alarm - onset_step`` in samples, ``0`` for an alarm on the
        onset sample itself, ``inf`` when the run ends with no alarm.

        This is an **index difference**, one less than the *run length* the
        expressions in :mod:`fdiscope.analytic` return.  Add one before
        comparing with :func:`fdiscope.analytic.cusum_delay_siegmund`.

    Notes
    -----
    Alarms *before* onset are false alarms and are ignored here by starting
    the search at ``onset_step``; the false-alarm rate is measured separately
    on fault-free runs.  Reporting a pre-onset alarm as a fast detection would
    be the single easiest way to fake a good result, so it is not done.
    """
    idx = first_alarm_index(alarm, start=int(onset_step), persistence=persistence)
    if idx < 0:
        return float("inf")
    return float(idx - int(onset_step))
