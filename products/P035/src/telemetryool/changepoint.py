"""Offline change-point detection for a mean shift, with a designed false-alarm rate.

The statistic is the standardised two-sample mean difference maximised over the
split point.  For a segment ``x[a:b]`` of length ``n = b - a`` and a candidate
split ``tau`` with ``a < tau < b``:

    ``D(tau) = |S_tau - (tau - a) / n * S_b| / (sigma * sqrt(m (n - m) / n))``

where ``S`` is the cumulative sum, ``m = tau - a``, and ``sigma`` is the known
(or supplied) noise standard deviation.  ``D(tau)`` is the absolute value of the
standardised difference between the two segment means; the test statistic is
``max_tau D(tau)`` and the estimated change point is the maximising ``tau``.
Under the no-change hypothesis with iid normal noise the statistic is the
maximum of a standardised Brownian-bridge-like process, which has no convenient
closed form, so the threshold is obtained by Monte Carlo at a stated simulation
size and reported with its own standard error -- not quoted from a table.

Multiple change points are found by binary segmentation: split at the maximiser
if the statistic exceeds the threshold, then recurse into both halves.  Binary
segmentation is greedy and is known to be inconsistent when change points are
close together relative to the segment length; that is a property of the method,
not of this implementation, and the alternative implementations are named in the
README.

References
----------
Hinkley, D. V. (1970). "Inference about the change-point in a sequence of random
    variables." *Biometrika* 57(1), 1-17.  The maximised standardised
    mean-difference statistic.
Scott, A. J. and Knott, M. (1974). "A Cluster Analysis Method for Grouping Means
    in the Analysis of Variance." *Biometrics* 30(3), 507-512.  The binary
    segmentation recursion.
Killick, R., Fearnhead, P. and Eckley, I. A. (2012). "Optimal Detection of
    Changepoints With a Linear Computational Cost." *Journal of the American
    Statistical Association* 107(500), 1590-1598.  PELT, the exact-penalised
    alternative, which the `ruptures` package implements and this module does
    not.
Fryzlewicz, P. (2014). "Wild binary segmentation for multiple change-point
    detection." *Annals of Statistics* 42(6), 2243-2281.  The randomised
    variant that repairs binary segmentation's greedy failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .calibration import estimate_rate

__all__ = [
    "SegmentStatistic",
    "max_mean_shift_statistic",
    "ChangePointThreshold",
    "calibrate_change_point_threshold",
    "detect_change_points",
]


@dataclass(frozen=True)
class SegmentStatistic:
    """Maximised standardised mean-shift statistic for one segment.

    Attributes
    ----------
    statistic
        ``max_tau D(tau)``, dimensionless (sigma units cancel).
    index
        Absolute index of the maximising split: the first sample of the second
        segment.
    start, stop
        Half-open bounds of the segment examined.
    """

    statistic: float
    index: int
    start: int
    stop: int


def max_mean_shift_statistic(
    x: NDArray[np.float64], sigma: float = 1.0, start: int = 0, stop: int | None = None
) -> SegmentStatistic:
    """Maximised standardised mean-shift statistic over all splits of ``x[start:stop]``.

    Parameters
    ----------
    x
        1-D series, engineering units.
    sigma
        Known noise standard deviation, same units as ``x``, > 0.
    start, stop
        Half-open segment bounds; ``stop=None`` means the end of ``x``.

    Returns
    -------
    SegmentStatistic
        With ``statistic = -inf`` and ``index = -1`` when the segment has fewer
        than two samples.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {arr.shape}")
    if float(sigma) <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}")
    stop = arr.size if stop is None else int(stop)
    if not (0 <= start < stop <= arr.size):
        raise ValueError(f"invalid segment bounds start={start} stop={stop} for n={arr.size}")
    seg = arr[start:stop]
    n = seg.size
    if n < 2:
        return SegmentStatistic(-np.inf, -1, int(start), int(stop))
    cs = np.cumsum(seg)
    m = np.arange(1, n, dtype=float)
    centred = cs[:-1] - m / n * cs[-1]
    scale = float(sigma) * np.sqrt(m * (n - m) / n)
    d = np.abs(centred) / scale
    j = int(np.argmax(d))
    return SegmentStatistic(float(d[j]), int(start + j + 1), int(start), int(stop))


@dataclass(frozen=True)
class ChangePointThreshold:
    """Monte-Carlo threshold for the change-point statistic under no change.

    Attributes
    ----------
    threshold
        Critical value of ``max_tau D(tau)``, dimensionless.
    alpha
        Target per-segment false-alarm probability.
    n_segments
        Segment length ``n`` the threshold was calibrated for.  The null
        distribution depends on ``n``, so a threshold is not transferable to a
        different length.
    n_simulations
        Monte-Carlo replicates used.
    achieved_alpha
        Fraction of replicates exceeding ``threshold``.
    standard_error
        Binomial standard error of ``achieved_alpha``.
    """

    threshold: float
    alpha: float
    n_segments: int
    n_simulations: int
    achieved_alpha: float
    standard_error: float


def calibrate_change_point_threshold(
    n_segments: int,
    alpha: float,
    n_simulations: int = 20000,
    rng: np.random.Generator | None = None,
) -> ChangePointThreshold:
    """Monte-Carlo critical value of the statistic under iid standard normal noise.

    Parameters
    ----------
    n_segments
        Segment length ``n``, >= 2.
    alpha
        Target per-segment false-alarm probability in (0, 1).
    n_simulations
        Replicates.  The relative standard error of the achieved rate is
        ``sqrt((1 - alpha) / (alpha * n_simulations))``: at ``alpha = 0.05`` and
        20000 replicates that is 3.1 %.
    rng
        Generator; ``None`` uses ``numpy.random.default_rng(0)`` so the result is
        reproducible by default.

    Returns
    -------
    ChangePointThreshold
    """
    if n_segments < 2:
        raise ValueError(f"n_segments must be >= 2, got {n_segments!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations!r}")
    if float(alpha) * n_simulations < 1.0:
        raise ValueError(
            f"alpha={alpha} needs at least {int(np.ceil(1.0 / float(alpha)))} simulations "
            f"to be representable, got {n_simulations}"
        )
    rng = np.random.default_rng(0) if rng is None else rng
    n = int(n_segments)
    draws = rng.standard_normal((int(n_simulations), n))
    cs = np.cumsum(draws, axis=1)
    m = np.arange(1, n, dtype=float)
    centred = cs[:, :-1] - (m / n)[None, :] * cs[:, -1][:, None]
    stat = (np.abs(centred) / np.sqrt(m * (n - m) / n)[None, :]).max(axis=1)
    k = int(np.ceil(float(alpha) * stat.size))
    cut = float(np.sort(stat)[::-1][k - 1])
    thresh = float(np.nextafter(cut, -np.inf))
    exceeded = int((stat > thresh).sum())
    est = estimate_rate(exceeded, stat.size)
    return ChangePointThreshold(
        threshold=thresh,
        alpha=float(alpha),
        n_segments=n,
        n_simulations=int(n_simulations),
        achieved_alpha=est.rate,
        standard_error=est.standard_error,
    )


def detect_change_points(
    x: NDArray[np.float64],
    threshold: float,
    sigma: float = 1.0,
    min_segment: int = 10,
    max_depth: int = 20,
) -> list[int]:
    """Binary-segmentation change-point indices, sorted ascending.

    Parameters
    ----------
    x
        1-D series, engineering units.
    threshold
        Critical value from :func:`calibrate_change_point_threshold`.  It is
        applied at every recursion level, so the *series-wide* false-alarm rate
        can exceed the per-segment ``alpha`` the threshold was calibrated for.
        The measured inflation at ``n = 200``, ``alpha = 0.05`` is only 1.03x,
        because the top-level test gates the recursion; see
        ``validation/validate_changepoint.py`` for the measurement at other
        settings.
    sigma
        Known noise standard deviation, same units as ``x``, > 0.
    min_segment
        Shortest segment that may be split further, >= 2.  Prevents the
        statistic being evaluated where it is dominated by its own variance.
    max_depth
        Recursion depth cap, >= 1, so a pathological series cannot recurse
        without bound.

    Returns
    -------
    list of int
        Indices of the first sample after each detected change.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {arr.shape}")
    if min_segment < 2:
        raise ValueError(f"min_segment must be >= 2, got {min_segment!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")
    found: list[int] = []
    stack = [(0, arr.size, 0)]
    while stack:
        a, b, depth = stack.pop()
        if b - a < 2 * min_segment or depth >= max_depth:
            continue
        st = max_mean_shift_statistic(arr, sigma, a, b)
        if st.statistic <= float(threshold) or st.index <= a or st.index >= b:
            continue
        found.append(st.index)
        stack.append((a, st.index, depth + 1))
        stack.append((st.index, b, depth + 1))
    return sorted(found)
