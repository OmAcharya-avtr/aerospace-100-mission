"""Host-to-device clock drift and reset detection.

Why a verdict needs this
------------------------
Every number in a timing verdict is a duration, and every duration was
measured by a clock. If the device clock runs 0.5 per cent fast, every
execution time in the log is 0.5 per cent too long and the set of iterations
that appear to have missed their deadline is not the set that actually did. If
the device clock **reset** part-way through the run, durations spanning the
reset are not merely biased, they are meaningless. So the clock is checked
before anything else is believed, and a run with an undetected reset is
reported as undecidable rather than analysed.

The model, and what is estimated
--------------------------------
The host series is the reference. Within a segment between resets the device
timestamps obey

```
device_k = o_j + (1 + rho)(host_k - host_{r_j}) + noise_k                  (20)
```

so ``rho`` is a slope minus one and is reported in parts per million. Two
estimators are provided and both are reported, because they fail differently:

- **Least squares** on ``(host, device)`` within a segment. Efficient under the
  Gaussian jitter of equation (19), and pulled by a single bad timestamp.
- **Theil-Sen**, the median of pairwise slopes (Theil 1950, *Nederlandse
  Akademie van Wetenschappen Proceedings* 53, 386-392 and 521-525; Sen 1968,
  "Estimates of the regression coefficient based on Kendall's tau", *JASA*
  63(324), 1379-1389). Breakdown point 29 per cent, so a handful of corrupted
  timestamps do not move it.

The reset detector
------------------
A reset shows up in the **offset series** ``residual_k = device_k - host_k`` as
a jump far larger than anything the drift and the jitter can produce in one
period. The threshold is built from the data rather than declared as an
absolute: the robust scale of the first differences, through the median
absolute deviation scaled by 1.4826 to be consistent for a Gaussian (Rousseeuw
& Croux 1993, "Alternatives to the median absolute deviation", *JASA*
88(424), 1273-1283, section 1), times a declared number of scales, with a
declared absolute floor so that a perfectly quantised series whose robust
scale is exactly zero does not make every difference a reset.

A detector with a data-driven threshold can be fooled in both directions, and
both are measured rather than argued about:
``validation/validate_clock.py`` reports detection recall and precision
against injected resets of known index, and the false-positive rate on
reset-free series, over a sweep of drift rates and jitter levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timeline import RunLog

__all__ = [
    "CLOCK_CLEAN_VERDICT",
    "CLOCK_DRIFTING",
    "CLOCK_RESET",
    "ClockEstimate",
    "ClockSegment",
    "detect_resets",
    "drift_corrected_log",
    "estimate_clock",
    "least_squares_rate",
    "theil_sen_rate",
]

#: The device clock tracks the host within the declared drift allowance.
CLOCK_CLEAN_VERDICT = "CLOCK_CLEAN"

#: The device clock has a rate error beyond the declared allowance. Durations
#: are biased and must be corrected before the timing verdict means anything.
CLOCK_DRIFTING = "CLOCK_DRIFTING"

#: The device clock reset during the run. Durations spanning a reset are not
#: biased, they are meaningless, and the run is undecidable.
CLOCK_RESET = "CLOCK_RESET"

#: Scale factor making the median absolute deviation consistent for a Gaussian.
MAD_TO_SIGMA = 1.4826


def least_squares_rate(
    host_time_s: np.ndarray, device_time_s: np.ndarray
) -> tuple[float, float]:
    """Least-squares slope of equation (20) and its standard error.

    Returns ``(rate, standard_error)``, both dimensionless. ``rate - 1`` times
    1e6 is the drift in parts per million. Needs at least three samples; with
    two the residual degrees of freedom are zero and the standard error does
    not exist, which is reported as ``nan`` rather than as zero.
    """
    host = np.asarray(host_time_s, dtype=float)
    device = np.asarray(device_time_s, dtype=float)
    if host.shape != device.shape:
        raise ValueError(
            f"host and device series must match, got {host.shape} and {device.shape}"
        )
    samples = int(host.shape[0])
    if samples < 2:
        raise ValueError(f"need at least 2 samples to estimate a rate, got {samples}")
    centred_host = host - host.mean()
    denominator = float(np.sum(centred_host**2))
    if denominator == 0.0:
        raise ValueError(
            "every host timestamp is identical, so the slope of equation (20) "
            "is not identified"
        )
    slope = float(np.sum(centred_host * (device - device.mean())) / denominator)
    if samples < 3:
        return slope, float("nan")
    intercept = float(device.mean() - slope * host.mean())
    residual = device - (intercept + slope * host)
    variance = float(np.sum(residual**2) / (samples - 2))
    return slope, float(np.sqrt(variance / denominator))


def theil_sen_rate(
    host_time_s: np.ndarray, device_time_s: np.ndarray, max_pairs: int = 40000
) -> float:
    """Theil-Sen slope: the median of pairwise slopes of equation (20).

    All pairs are used when there are at most ``max_pairs`` of them; above
    that, a deterministic stride over the pair grid is taken so the result is
    reproducible and the cost is bounded. Dimensionless.
    """
    host = np.asarray(host_time_s, dtype=float)
    device = np.asarray(device_time_s, dtype=float)
    if host.shape != device.shape:
        raise ValueError(
            f"host and device series must match, got {host.shape} and {device.shape}"
        )
    samples = int(host.shape[0])
    if samples < 2:
        raise ValueError(f"need at least 2 samples for Theil-Sen, got {samples}")
    rows, columns = np.triu_indices(samples, k=1)
    total = int(rows.shape[0])
    if total > max_pairs:
        stride = int(np.ceil(total / max_pairs))
        rows = rows[::stride]
        columns = columns[::stride]
    spacing = host[columns] - host[rows]
    usable = spacing != 0.0
    if not np.any(usable):
        raise ValueError("no pair of host timestamps differs; the slope is not identified")
    slopes = (device[columns][usable] - device[rows][usable]) / spacing[usable]
    return float(np.median(slopes))


def detect_resets(
    host_time_s: np.ndarray,
    device_time_s: np.ndarray,
    scales: float = 8.0,
    floor_s: float = 1.0e-4,
) -> tuple[int, ...]:
    """Indices at which the device clock reset, from the offset series.

    Parameters
    ----------
    host_time_s, device_time_s
        Paired timestamp series, s.
    scales
        How many robust scales of the first differences a jump must exceed.
        Declared; its effect on recall and precision is swept in validation.
    floor_s
        Absolute floor on the threshold, s. Present because a perfectly
        quantised series can have a robust scale of exactly zero, and then a
        purely multiplicative threshold calls every tick boundary a reset.

    Returns
    -------
    tuple of int
        Sample indices at which a reset takes effect: index ``r`` means the
        device clock restarted at sample ``r``, so segment boundaries are
        ``[0, r)`` and ``[r, ...)``.
    """
    if scales <= 0.0:
        raise ValueError(f"scales must be positive, got {scales}")
    if floor_s <= 0.0:
        raise ValueError(f"floor_s must be positive, got {floor_s}")
    host = np.asarray(host_time_s, dtype=float)
    device = np.asarray(device_time_s, dtype=float)
    if host.shape != device.shape:
        raise ValueError(
            f"host and device series must match, got {host.shape} and {device.shape}"
        )
    if host.shape[0] < 3:
        return ()
    residual = device - host
    differences = np.diff(residual)
    median = float(np.median(differences))
    deviation = float(np.median(np.abs(differences - median)))
    scale = MAD_TO_SIGMA * deviation
    threshold = max(scales * scale, floor_s)
    jumps = np.nonzero(np.abs(differences - median) > threshold)[0]
    return tuple(int(index) + 1 for index in jumps)


@dataclass(frozen=True)
class ClockSegment:
    """One stretch of the run between resets.

    Attributes
    ----------
    start, stop
        Half-open sample index range.
    drift_ppm
        Least-squares drift of equation (20), parts per million.
    drift_ppm_standard_error
        Standard error of ``drift_ppm``, parts per million; ``nan`` for a
        segment with fewer than three samples.
    drift_ppm_theil_sen
        Theil-Sen drift, parts per million.
    """

    start: int
    stop: int
    drift_ppm: float
    drift_ppm_standard_error: float
    drift_ppm_theil_sen: float

    @property
    def samples(self) -> int:
        """Number of timestamp pairs in the segment."""
        return self.stop - self.start


@dataclass(frozen=True)
class ClockEstimate:
    """What the clock check found.

    Attributes
    ----------
    segments
        One :class:`ClockSegment` per stretch between resets.
    resets
        Detected reset indices.
    drift_ppm
        Sample-weighted mean of the segment drifts, parts per million. The
        pooled number a report quotes.
    drift_ppm_standard_error
        Pooled standard error, parts per million, combining the segment
        standard errors in inverse-variance weighting. ``nan`` when no segment
        has one.
    verdict
        :data:`CLOCK_CLEAN_VERDICT`, :data:`CLOCK_DRIFTING` or
        :data:`CLOCK_RESET`.
    allowance_ppm
        The declared drift allowance the verdict was taken against.
    """

    segments: tuple[ClockSegment, ...]
    resets: tuple[int, ...]
    drift_ppm: float
    drift_ppm_standard_error: float
    verdict: str
    allowance_ppm: float

    @property
    def usable(self) -> bool:
        """True when the timing trace can be believed after correction."""
        return self.verdict != CLOCK_RESET


def estimate_clock(
    host_time_s: np.ndarray,
    device_time_s: np.ndarray,
    allowance_ppm: float = 100.0,
    scales: float = 8.0,
    floor_s: float = 1.0e-4,
) -> ClockEstimate:
    """Detect resets, fit a rate per segment, and give the clock a verdict.

    The verdict is :data:`CLOCK_RESET` whenever a reset is detected, whatever
    the fitted rates look like, because a reset inside the run breaks the
    durations and no rate correction repairs that.
    """
    if allowance_ppm < 0.0:
        raise ValueError(f"allowance_ppm must be non-negative, got {allowance_ppm}")
    host = np.asarray(host_time_s, dtype=float)
    device = np.asarray(device_time_s, dtype=float)
    resets = detect_resets(host, device, scales=scales, floor_s=floor_s)
    boundaries = (0, *resets, int(host.shape[0]))
    segments: list[ClockSegment] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop - start < 2:
            continue
        rate, error = least_squares_rate(host[start:stop], device[start:stop])
        robust = theil_sen_rate(host[start:stop], device[start:stop])
        segments.append(
            ClockSegment(
                start=int(start),
                stop=int(stop),
                drift_ppm=(rate - 1.0) * 1.0e6,
                drift_ppm_standard_error=error * 1.0e6,
                drift_ppm_theil_sen=(robust - 1.0) * 1.0e6,
            )
        )
    if not segments:
        raise ValueError(
            "no segment between resets has two or more samples, so no rate is "
            "identified; the timestamp series cannot be checked"
        )
    weights = np.array([segment.samples for segment in segments], dtype=float)
    drifts = np.array([segment.drift_ppm for segment in segments], dtype=float)
    pooled = float(np.sum(weights * drifts) / np.sum(weights))
    errors = np.array(
        [segment.drift_ppm_standard_error for segment in segments], dtype=float
    )
    finite = np.isfinite(errors) & (errors > 0.0)
    if np.any(finite):
        pooled_error = float(np.sqrt(1.0 / np.sum(1.0 / errors[finite] ** 2)))
    else:
        pooled_error = float("nan")
    if resets:
        verdict = CLOCK_RESET
    elif abs(pooled) > allowance_ppm:
        verdict = CLOCK_DRIFTING
    else:
        verdict = CLOCK_CLEAN_VERDICT
    return ClockEstimate(
        segments=tuple(segments),
        resets=resets,
        drift_ppm=pooled,
        drift_ppm_standard_error=pooled_error,
        verdict=verdict,
        allowance_ppm=float(allowance_ppm),
    )


def drift_corrected_log(log: RunLog, drift_ppm: float) -> RunLog:
    """A copy of ``log`` with the device-measured durations divided by ``1 + rho``.

    A duration measured on a clock running fast by ``rho`` reads ``1 + rho``
    times the true interval, so the correction is a division. This changes
    which iterations appear to have missed their deadline, and that change is
    measured in ``validation/validate_clock.py`` rather than assumed to be
    negligible.
    """
    rate = 1.0 + float(drift_ppm) * 1.0e-6
    if rate <= 0.0:
        raise ValueError(
            f"a drift of {drift_ppm} ppm implies a non-positive clock rate; a "
            "duration cannot be corrected by it"
        )
    corrected = np.asarray(log.durations_s, dtype=float) / rate
    return RunLog(
        model=log.model,
        command_rad=log.command_rad,
        durations_s=corrected,
        states=log.states,
        computed_rad=log.computed_rad,
        saturated=log.saturated,
        host_time_s=log.host_time_s,
        device_time_s=log.device_time_s,
        prior_command_rad=log.prior_command_rad,
        label=f"{log.label}+drift-corrected",
    )
