"""A streaming monitor: one detector, one declared threshold, one verdict.

This is the object a user holds. It carries a detector, the threshold that was
set from a declared false-alarm target, and nothing else -- in particular it
carries no ability to say *why* the twin and the asset disagree. See
:mod:`twininvalidate.ambiguity`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arl import ArlEstimate, arl0_estimate, arl1_estimate
from .detectors import DetectorSpec, first_alarm
from .thresholds import Calibration, calibrate_threshold, rate_from_arl0


@dataclass(frozen=True)
class MonitorVerdict:
    """The outcome of running a monitor over one or more residual streams.

    Attributes
    ----------
    alarm_index:
        Per-run index of the first alarm, ``-1`` where no alarm occurred,
        shape ``(n_runs,)``.
    statistic:
        The statistic path, shape ``(n_runs, n_samples)``.
    threshold:
        The threshold used.
    """

    alarm_index: np.ndarray
    statistic: np.ndarray
    threshold: float

    @property
    def alarmed(self) -> np.ndarray:
        """Boolean mask of runs that alarmed."""
        return self.alarm_index >= 0

    @property
    def alarm_fraction(self) -> float:
        """Fraction of runs that alarmed."""
        return float(self.alarmed.mean())

    def describe(self) -> str:
        """One-line human-readable summary. Returned, never printed."""
        n = int(self.alarmed.sum())
        total = int(self.alarm_index.size)
        if n == 0:
            return (
                f"no alarm in {total} run(s) over {self.statistic.shape[1]} samples "
                f"at threshold {self.threshold:.4g}"
            )
        first = int(self.alarm_index[self.alarmed].min())
        mean = float(self.alarm_index[self.alarmed].mean() + 1)
        return (
            f"{n}/{total} run(s) alarmed at threshold {self.threshold:.4g}; "
            f"earliest at sample {first}, mean alarm index {mean:.1f} samples"
        )


@dataclass(frozen=True)
class InvalidationMonitor:
    """A detector plus the threshold that a declared false-alarm target bought.

    Build one with :meth:`from_target`, which is the only constructor that can
    guarantee the threshold was set without looking at out-of-control data.

    Attributes
    ----------
    spec:
        The detector and its declared design constants.
    threshold:
        Alarm threshold in the statistic's units.
    calibration:
        The :class:`twininvalidate.thresholds.Calibration` record, or ``None``
        if the threshold was supplied directly.
    sample_rate_hz:
        Monitor sample rate, used only to express false-alarm rates in hours.
    """

    spec: DetectorSpec
    threshold: float
    calibration: Calibration | None = None
    sample_rate_hz: float = 20.0

    @classmethod
    def from_target(
        cls,
        spec: DetectorSpec,
        in_control: np.ndarray,
        target_arl0: float,
        sample_rate_hz: float = 20.0,
    ) -> InvalidationMonitor:
        """Calibrate ``spec``'s threshold on in-control data and return a monitor.

        Parameters
        ----------
        spec:
            Detector and declared design constants.
        in_control:
            In-control normalised residual streams, shape
            ``(n_runs, n_samples)``.
        target_arl0:
            Declared target in-control average run length, in samples.
        sample_rate_hz:
            Monitor sample rate in Hz.
        """
        cal = calibrate_threshold(spec, in_control, target_arl0)
        return cls(
            spec=spec,
            threshold=cal.threshold,
            calibration=cal,
            sample_rate_hz=float(sample_rate_hz),
        )

    def run(self, z: np.ndarray) -> MonitorVerdict:
        """Run the monitor over residual streams.

        Parameters
        ----------
        z:
            Normalised residuals, shape ``(n_runs, n_samples)`` or
            ``(n_samples,)``, dimensionless.
        """
        stat = self.spec.statistic(z)
        return MonitorVerdict(
            alarm_index=first_alarm(stat, self.threshold),
            statistic=stat,
            threshold=self.threshold,
        )

    def arl0(self, in_control: np.ndarray) -> ArlEstimate:
        """Censored-MLE in-control ARL0 on fresh in-control streams, samples."""
        return arl0_estimate(self.spec.statistic(in_control), self.threshold)

    def arl1(self, out_of_control: np.ndarray) -> ArlEstimate:
        """Zero-state detection delay on streams changed from sample 0, samples."""
        return arl1_estimate(self.spec.statistic(out_of_control), self.threshold)

    def false_alarms_per_1000h(self, arl0_samples: float) -> float:
        """Convert an ARL0 in samples to false alarms per 1000 operating hours."""
        return rate_from_arl0(arl0_samples, self.sample_rate_hz)
