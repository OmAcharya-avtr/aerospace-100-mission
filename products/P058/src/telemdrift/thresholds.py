"""Threshold calibration: put every detector at the same measured ARL0.

The whole argument of this package is in this module. Five detectors with five
incomparable scalars -- a CUSUM decision interval, a Page-Hinkley lambda, an
EWMA control limit, a KS critical value and an ADWIN delta -- cannot be compared
at their defaults, because the defaults encode five different and mostly
unstated false-alarm rates. They become comparable only after each scalar is
moved until the *measured* ARL0 matches a declared target.

Procedure, stated once and used everywhere
------------------------------------------
1. Declare a target ARL0 in samples (the benchmark uses 500).
2. Bracket the threshold: start from the detector's declared default and expand
   geometrically outward until one end gives a measured ARL0 below target and
   the other above. ADWIN's ``delta`` is *inversely* related to ARL0, which the
   bracketing handles by detecting the sign of the relationship rather than
   assuming it.
3. Bisect on the monotone-in-expectation relationship for a fixed number of
   iterations, measuring ARL0 on **calibration seeds**.
4. Report the achieved ARL0 on **evaluation seeds that were not used for
   calibration**, with its Monte Carlo standard error.

Step 4 is not decoration. Bisecting against a noisy objective selects a
threshold whose calibration-set ARL0 is close to target partly by luck; the
honest number is the one measured afterwards on fresh seeds. The gap between
the two is measured and published in README.md.

Known non-monotonicity
----------------------
Measured ARL0 is monotone in each threshold only in expectation. At a fixed
seed set the measured curve is a noisy step function, so bisection can terminate
at a threshold whose neighbours both measure closer to target. The calibration
is therefore specified as a fixed iteration count with a reported achieved ARL0,
not as a convergence to a tolerance, because a tolerance that is smaller than
the Monte Carlo noise cannot be met and claiming it would be false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoring import ARL0Result, measure_arl0

__all__ = ["CalibrationResult", "calibrate_threshold"]


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one threshold calibration."""

    detector: str
    threshold: float
    target_arl0: float
    calibration_arl0: float
    achieved: ARL0Result
    iterations: int
    bracket: tuple[float, float]
    bracketing_failed: bool

    @property
    def achieved_arl0(self) -> float:
        return self.achieved.arl0

    @property
    def target_error(self) -> float:
        """``(achieved - target) / target``, dimensionless and signed."""
        return (self.achieved.arl0 - self.target_arl0) / self.target_arl0

    def summary(self) -> str:
        warn = "  [BRACKETING FAILED]" if self.bracketing_failed else ""
        return (
            f"{self.detector:14s} threshold={self.threshold:.6g}  "
            f"calibration ARL0={self.calibration_arl0:8.1f}  "
            f"held-out {self.achieved.summary()}  "
            f"target error {100 * self.target_error:+.1f} %{warn}"
        )


def _arl0_at(
    factory_from_threshold,
    threshold: float,
    stream_fn,
    seeds,
    stream_length: int,
) -> float:
    return measure_arl0(
        lambda: factory_from_threshold(threshold), stream_fn, seeds, stream_length
    ).arl0


def calibrate_threshold(
    name: str,
    factory_from_threshold,
    default_threshold: float,
    stream_fn,
    target_arl0: float,
    calibration_seeds,
    evaluation_seeds,
    calibration_length: int = 25_000,
    evaluation_length: int = 50_000,
    iterations: int = 14,
    bracket_steps: int = 26,
    bracket_factor: float = 1.6,
    clip: tuple[float, float] | None = None,
) -> CalibrationResult:
    """Move one scalar until the measured ARL0 matches ``target_arl0``.

    Parameters
    ----------
    name:
        Detector label for reports.
    factory_from_threshold:
        ``f(threshold) -> Detector``.
    default_threshold:
        Starting point for the bracket search.
    stream_fn:
        ``f(length, seed) -> np.ndarray`` stationary stream generator.
    target_arl0:
        Target mean samples to false alarm, > 0.
    calibration_seeds, evaluation_seeds:
        Disjoint seed sets. Overlap is rejected, because an achieved ARL0
        measured on the seeds it was fitted to is not an achieved ARL0.
    clip:
        Optional ``(lo, hi)`` hard bounds on the threshold, needed for ADWIN
        whose ``delta`` must stay in ``(0, 1)``.

    Returns
    -------
    CalibrationResult
        Including ``bracketing_failed``, which is ``True`` when the target ARL0
        could not be bracketed within ``bracket_steps``. The result is still
        returned with the best endpoint rather than raising, so a sweep over
        many targets records the failure instead of aborting.
    """
    if target_arl0 <= 0:
        raise ValueError("target_arl0 must be > 0")
    cal = list(calibration_seeds)
    ev = list(evaluation_seeds)
    if set(cal) & set(ev):
        raise ValueError("calibration_seeds and evaluation_seeds must be disjoint")
    if not cal or not ev:
        raise ValueError("both seed sets must be non-empty")

    def arl0(th: float) -> float:
        return _arl0_at(factory_from_threshold, th, stream_fn, cal, calibration_length)

    lo_clip, hi_clip = clip if clip is not None else (1e-12, 1e12)

    def clamp(v: float) -> float:
        return min(max(v, lo_clip), hi_clip)

    th0 = clamp(default_threshold)
    a0 = arl0(th0)
    # Determine the sign of d(ARL0)/d(threshold) empirically rather than by
    # assumption: it is positive for CUSUM/PH/EWMA/KS and negative for ADWIN.
    th_probe = clamp(th0 * bracket_factor)
    a_probe = arl0(th_probe)
    increasing = a_probe >= a0

    lo, hi = th0, th0
    a_lo, a_hi = a0, a0
    bracketing_failed = True
    for _ in range(bracket_steps):
        below = a_lo < target_arl0 if increasing else a_lo > target_arl0
        above = a_hi > target_arl0 if increasing else a_hi < target_arl0
        if below and above:
            bracketing_failed = False
            break
        if not below:
            lo = clamp(lo / bracket_factor)
            a_lo = arl0(lo)
        if not above:
            hi = clamp(hi * bracket_factor)
            a_hi = arl0(hi)
        if lo <= lo_clip and hi >= hi_clip:
            break
    else:
        below = a_lo < target_arl0 if increasing else a_lo > target_arl0
        above = a_hi > target_arl0 if increasing else a_hi < target_arl0
        bracketing_failed = not (below and above)

    it_done = 0
    if not bracketing_failed:
        for _ in range(iterations):
            mid = float(np.sqrt(lo * hi)) if lo > 0 and hi > 0 else 0.5 * (lo + hi)
            mid = clamp(mid)
            a_mid = arl0(mid)
            it_done += 1
            if (a_mid < target_arl0) == increasing:
                lo, a_lo = mid, a_mid
            else:
                hi, a_hi = mid, a_mid
        chosen = float(np.sqrt(lo * hi)) if lo > 0 and hi > 0 else 0.5 * (lo + hi)
    else:
        # Keep whichever endpoint measured closest to the target.
        chosen = lo if abs(a_lo - target_arl0) <= abs(a_hi - target_arl0) else hi
    chosen = clamp(chosen)
    cal_arl0 = arl0(chosen)
    achieved = measure_arl0(
        lambda: factory_from_threshold(chosen), stream_fn, ev, evaluation_length
    )
    return CalibrationResult(
        detector=name,
        threshold=chosen,
        target_arl0=float(target_arl0),
        calibration_arl0=cal_arl0,
        achieved=achieved,
        iterations=it_done,
        bracket=(lo, hi),
        bracketing_failed=bracketing_failed,
    )
