"""The classical speed-scaling predictor. Fitted to nothing.

With ``m`` cores and ``n`` busy co-resident workers there are ``n+1`` runnable
entities competing for ``m`` cores, so under an ideal fair scheduler each
receives

    available_fraction = min(1, m / (n + 1))

of one core's throughput, and a job whose execution time on an otherwise-idle
board is ``e`` is predicted to take ``e / available_fraction``. The predicted
miss probability is then the fraction of the **idle-run** execution-time
distribution whose scaled value exceeds the deadline.

This is the textbook capacity argument -- the processor-sharing limit of a
generalised processor-sharing scheduler, as in Kleinrock 1976, "Queueing
Systems, Volume 2: Computer Applications", chapter 4, and the uniform
multiprocessor speed-scaling model used in real-time scheduling analysis, e.g.
Funk, Goossens & Baruah 2001, "On-line Scheduling on Uniform Multiprocessors",
RTSS 2001. It uses only descriptors an operator can observe without running the
task, plus the idle-run execution-time distribution.

Assumptions, which are the reason it is a baseline and not an oracle: ideal fair
sharing, no cache or memory-bandwidth interference, no frequency scaling, no
scheduling-latency component, and workers that are continuously runnable. Every
one of those is false on a real SoC in a direction that makes this an
optimistic predictor. Validity range: ``m >= 1``, ``n >= 0``.

Units: execution times and deadlines in seconds; fractions and probabilities
dimensionless.
"""

from __future__ import annotations

import numpy as np

#: Cores on the measured board. Declared here so a caller cannot pass the wrong
#: count silently; the campaign records it in the manifest as ``env.nproc``.
BOARD_CORES = 6


def available_fraction(cores: int, n_workers: int) -> float:
    """``min(1, cores / (n_workers + 1))``, dimensionless.

    Raises
    ------
    ValueError
        On ``cores < 1`` or ``n_workers < 0``.
    """
    if cores < 1:
        raise ValueError(f"cores must be >= 1, got {cores}")
    if n_workers < 0:
        raise ValueError(f"n_workers must be >= 0, got {n_workers}")
    return float(min(1.0, cores / (n_workers + 1.0)))


def predicted_response(
    idle_exec_s: np.ndarray, cores: int, n_workers: int
) -> np.ndarray:
    """Idle execution times scaled by the inverse available fraction, seconds."""
    values = np.asarray(idle_exec_s, dtype=np.float64)
    if values.size == 0:
        raise ValueError("idle_exec_s is empty")
    if np.any(values < 0.0):
        raise ValueError("idle_exec_s contains a negative value")
    return values / available_fraction(cores, n_workers)


def predicted_miss_probability(
    idle_exec_s: np.ndarray, deadline_s: float, cores: int, n_workers: int
) -> float:
    """Fraction of the scaled idle distribution exceeding ``deadline_s``.

    Raises
    ------
    ValueError
        On an empty idle sample or a non-positive deadline.
    """
    if not np.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError(f"deadline_s must be finite and > 0, got {deadline_s!r}")
    scaled = predicted_response(idle_exec_s, cores, n_workers)
    return float(np.count_nonzero(scaled > deadline_s) / scaled.size)
