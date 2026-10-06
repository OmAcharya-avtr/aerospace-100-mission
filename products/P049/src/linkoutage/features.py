"""Observable-history features and labels for short-horizon outage prediction.

The prediction problem
----------------------
At decision instant ``t`` the receiver has observed ``a[t-W+1 .. t]`` and
nothing later. The question is whether a **new** fade begins in
``(t, t + H]`` -- that is, whether the amplitude performs a down-crossing of
the outage threshold within the next ``H`` samples. A fade already in progress
at ``t`` is not a positive label by itself; it is already known to the
receiver and predicting it would be worthless.

Everything in this module exists to make that question answerable without
leakage:

* features are computed from ``a[t-W+1 .. t]`` only,
* labels are computed from ``a[t+1 .. t+H]`` only,
* decision instants are spaced by a stride, and the train / calibration / test
  split is **temporal with a gap**, never random. A random split of a
  correlated series leaks: two decision instants a few samples apart share
  almost all of their window and most of their horizon, so a shuffled split
  puts near-duplicates of test rows into the training set and inflates every
  metric.

The effective sample size is set by the number of independent fade events in
the record, not by the number of rows. At a 50-sample stride and a 200-sample
horizon, consecutive rows overlap heavily; the row count is therefore not a
measure of statistical power, and :func:`temporal_split` reports the number of
labelled onsets per split for that reason.

Units
-----
Amplitude in arbitrary linear units; windows, strides, horizons and
``samples_since_down_crossing`` in samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fade import DEFAULT_DEFINITIONS, FadeDefinitions, down_crossing_indices

__all__ = [
    "FEATURE_NAMES",
    "OutageDataset",
    "SplitIndices",
    "build_outage_dataset",
    "decision_indices",
    "extract_features",
    "onset_labels",
    "temporal_split",
]

FEATURE_NAMES: tuple[str, ...] = (
    "log_amp_last",
    "log_amp_mean",
    "log_amp_std",
    "log_amp_min",
    "log_amp_max",
    "log_amp_p10",
    "log_amp_slope_full",
    "log_amp_slope_short",
    "log_amp_mean_short",
    "log_amp_diff_std",
    "frac_below_threshold",
    "n_down_crossings_in_window",
    "samples_since_down_crossing",
    "in_fade_now",
)
"""The 14 features, in column order.

``log_amp_*``
    Natural log of amplitude. Taken in the log domain because the model is
    lognormal there, so a Gaussian-shaped feature distribution is the one the
    linear baseline can actually use.
``log_amp_slope_full`` / ``log_amp_slope_short``
    Ordinary-least-squares slope per sample over the whole window and over its
    last eighth.
``log_amp_diff_std``
    Standard deviation of the first difference of log amplitude: a local
    roughness measure that distinguishes a slow deep excursion from a noisy
    shallow one.
``frac_below_threshold`` / ``n_down_crossings_in_window``
    How much of the recent past was already in fade, and how often it entered.
``samples_since_down_crossing``
    Samples since the most recent down-crossing inside the window, capped at
    the window length when there is none.
``in_fade_now``
    1.0 when ``a[t]`` is below the threshold.
"""


def decision_indices(
    n_samples: int, *, window_samples: int, horizon_samples: int, stride_samples: int
) -> NDArray[np.int64]:
    """Valid decision instants ``t``.

    A decision instant needs a full window behind it and a full horizon ahead:
    ``window_samples - 1 <= t <= n_samples - 1 - horizon_samples``.
    """
    n = int(n_samples)
    w = int(window_samples)
    h = int(horizon_samples)
    st = int(stride_samples)
    if w < 2:
        raise ValueError(f"window_samples must be >= 2, got {window_samples!r}")
    if h < 1:
        raise ValueError(f"horizon_samples must be >= 1, got {horizon_samples!r}")
    if st < 1:
        raise ValueError(f"stride_samples must be >= 1, got {stride_samples!r}")
    first = w - 1
    last = n - 1 - h
    if last < first:
        raise ValueError(
            f"n_samples={n} is too short for window_samples={w} and horizon_samples={h}; "
            f"need at least {w + h} samples"
        )
    return np.arange(first, last + 1, st, dtype=np.int64)


def extract_features(
    amplitude: ArrayLike,
    *,
    index: ArrayLike,
    window_samples: int,
    threshold: float,
    chunk: int = 4096,
) -> NDArray[np.float64]:
    """Feature matrix, one row per decision instant.

    Parameters
    ----------
    amplitude:
        Full amplitude record, strictly positive (the features are logarithmic).
    index:
        Decision instants ``t``; every one needs ``t >= window_samples - 1``.
    window_samples:
        Window length ``W``.
    threshold:
        Outage threshold, same units as ``amplitude``.
    chunk:
        Rows per internal block. Bounds peak memory at
        ``chunk * window_samples`` float64, so 4096 x 400 is about 13 MB.

    Returns
    -------
    ndarray, shape ``(len(index), 14)``
        Columns in the order of :data:`FEATURE_NAMES`.
    """
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError(f"amplitude must have at least 2 samples, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError("amplitude contains non-finite values")
    if np.any(a <= 0.0):
        raise ValueError(
            "amplitude must be strictly positive: the features are logarithmic. "
            "Add a floor, or pass irradiance with a matching squared threshold."
        )
    idx = np.asarray(index, dtype=np.int64).ravel()
    w = int(window_samples)
    if w < 2:
        raise ValueError(f"window_samples must be >= 2, got {window_samples!r}")
    if idx.size == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    if idx.min() < w - 1:
        raise ValueError(
            f"index contains {int(idx.min())}, which has fewer than window_samples={w} "
            "samples of history behind it"
        )
    if idx.max() >= a.size:
        raise ValueError(f"index contains {int(idx.max())} >= n_samples={a.size}")
    t = float(threshold)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"threshold must be positive and finite, got {threshold!r}")
    ch = int(chunk)
    if ch < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk!r}")

    log_a = np.log(a)
    log_t = float(np.log(t))
    short = max(2, w // 8)

    pos = np.arange(w, dtype=np.float64)
    pos_c = pos - pos.mean()
    slope_w = pos_c / float(np.sum(pos_c * pos_c))
    pos_s = np.arange(short, dtype=np.float64)
    pos_sc = pos_s - pos_s.mean()
    slope_s = pos_sc / float(np.sum(pos_sc * pos_sc))

    out = np.empty((idx.size, len(FEATURE_NAMES)), dtype=np.float64)
    for lo in range(0, idx.size, ch):
        hi = min(lo + ch, idx.size)
        sel = idx[lo:hi]
        offsets = sel[:, None] - (w - 1) + np.arange(w, dtype=np.int64)[None, :]
        win = log_a[offsets]
        below = win < log_t
        block = out[lo:hi]
        block[:, 0] = win[:, -1]
        block[:, 1] = win.mean(axis=1)
        block[:, 2] = win.std(axis=1)
        block[:, 3] = win.min(axis=1)
        block[:, 4] = win.max(axis=1)
        block[:, 5] = np.quantile(win, 0.10, axis=1)
        block[:, 6] = win @ slope_w
        block[:, 7] = win[:, -short:] @ slope_s
        block[:, 8] = win[:, -short:].mean(axis=1)
        block[:, 9] = np.diff(win, axis=1).std(axis=1)
        block[:, 10] = below.mean(axis=1)
        enters = below[:, 1:] & (~below[:, :-1])
        block[:, 11] = enters.sum(axis=1)
        # Samples since the most recent in-window down-crossing; w when none.
        any_enter = enters.any(axis=1)
        last_enter = (w - 2) - np.argmax(enters[:, ::-1], axis=1)
        since = (w - 1) - (last_enter + 1)
        block[:, 12] = np.where(any_enter, since.astype(np.float64), float(w))
        block[:, 13] = below[:, -1].astype(np.float64)
    return out


def onset_labels(
    amplitude: ArrayLike,
    *,
    index: ArrayLike,
    threshold: float,
    horizon_samples: int,
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> NDArray[np.int64]:
    """1 when a down-crossing occurs in ``(t, t + H]``, else 0.

    The crossing definition is the one in :mod:`linkoutage.fade`, so the label
    and the measured level-crossing rate count the same events. In particular a
    single-sample excursion is an onset under the default definitions.
    """
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    idx = np.asarray(index, dtype=np.int64).ravel()
    h = int(horizon_samples)
    if h < 1:
        raise ValueError(f"horizon_samples must be >= 1, got {horizon_samples!r}")
    if idx.size and idx.max() + h >= a.size + 1:
        raise ValueError(
            f"index contains {int(idx.max())}, whose horizon of {h} samples runs past "
            f"the end of a record of {a.size} samples"
        )
    crossings = down_crossing_indices(a, threshold, definitions=definitions)
    if crossings.size == 0:
        return np.zeros(idx.size, dtype=np.int64)
    lo = np.searchsorted(crossings, idx, side="right")
    hi = np.searchsorted(crossings, idx + h, side="right")
    return (hi > lo).astype(np.int64)


@dataclass(frozen=True)
class SplitIndices:
    """Temporal train / calibration / test row ranges with gaps.

    Attributes
    ----------
    train, calibration, test:
        Row index arrays into the feature matrix, in time order.
    gap_rows:
        Rows discarded between consecutive splits.
    n_positive:
        Positives in each split, as ``(train, calibration, test)``. This, not
        the row count, is what limits what the metrics can resolve.
    """

    train: NDArray[np.int64]
    calibration: NDArray[np.int64]
    test: NDArray[np.int64]
    gap_rows: int
    n_positive: tuple[int, int, int]

    def report(self) -> str:
        return "\n".join(
            [
                "Temporal split (no shuffling; gap between splits to break overlap):",
                f"  train       : {self.train.size} rows, {self.n_positive[0]} positive",
                f"  calibration : {self.calibration.size} rows, {self.n_positive[1]} positive",
                f"  test        : {self.test.size} rows, {self.n_positive[2]} positive",
                f"  gap         : {self.gap_rows} rows discarded between splits",
            ]
        )


def temporal_split(
    y: ArrayLike,
    *,
    fractions: tuple[float, float, float] = (0.60, 0.15, 0.25),
    gap_rows: int = 0,
) -> SplitIndices:
    """Split rows in time order, discarding ``gap_rows`` between splits.

    ``gap_rows`` must be at least ``ceil((window + horizon) / stride)`` for the
    windows and horizons either side of a boundary to be disjoint. Callers that
    use :func:`build_outage_dataset` get this computed for them.
    """
    labels = np.asarray(y, dtype=np.int64).ravel()
    n = labels.size
    if n < 10:
        raise ValueError(f"need at least 10 rows to split, got {n}")
    f = tuple(float(v) for v in fractions)
    if len(f) != 3 or any(v <= 0.0 for v in f):
        raise ValueError(f"fractions must be three positive numbers, got {fractions!r}")
    if abs(sum(f) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {sum(f)!r}")
    g = int(gap_rows)
    if g < 0:
        raise ValueError(f"gap_rows must be >= 0, got {gap_rows!r}")
    n_train = int(round(f[0] * n))
    n_cal = int(round(f[1] * n))
    train = np.arange(0, n_train, dtype=np.int64)
    cal_start = n_train + g
    cal = np.arange(cal_start, min(cal_start + n_cal, n), dtype=np.int64)
    test_start = (cal[-1] + 1 + g) if cal.size else cal_start + g
    test = np.arange(test_start, n, dtype=np.int64)
    if train.size == 0 or cal.size == 0 or test.size == 0:
        raise ValueError(
            f"gap_rows={g} leaves an empty split for n={n} rows and fractions={f}"
        )
    return SplitIndices(
        train=train,
        calibration=cal,
        test=test,
        gap_rows=g,
        n_positive=(
            int(labels[train].sum()),
            int(labels[cal].sum()),
            int(labels[test].sum()),
        ),
    )


@dataclass(frozen=True)
class OutageDataset:
    """A complete, leakage-controlled short-horizon outage dataset.

    Attributes
    ----------
    x:
        Feature matrix, ``(n_rows, 14)``, columns per :data:`FEATURE_NAMES`.
    y:
        0/1 onset labels.
    index:
        Decision instant of each row, in samples into the source record.
    gaussian:
        The AR(1) Gaussian level ``x[t]`` at each decision instant, when the
        record came from :func:`linkoutage.channel.lognormal_amplitude_series`.
        The analytic predictor needs it; a user with real measurements gets
        ``None`` here and must use the measurement-driven baselines.
    split:
        Temporal split with the gap already sized.
    base_rate:
        Overall positive fraction.
    configuration:
        Everything needed to regenerate the dataset.
    """

    x: NDArray[np.float64]
    y: NDArray[np.int64]
    index: NDArray[np.int64]
    gaussian: NDArray[np.float64] | None
    split: SplitIndices
    base_rate: float
    configuration: dict[str, float | int | str]

    @property
    def n_rows(self) -> int:
        return int(self.y.size)

    def report(self) -> str:
        lines = ["Outage dataset:"]
        for key in sorted(self.configuration):
            lines.append(f"  {key:<24s}: {self.configuration[key]!r}")
        lines.append(f"  {'rows':<24s}: {self.n_rows}")
        lines.append(f"  {'positives':<24s}: {int(self.y.sum())}")
        lines.append(f"  {'base rate':<24s}: {self.base_rate!r}")
        lines.append("")
        lines.append(self.split.report())
        return "\n".join(lines)


def build_outage_dataset(
    amplitude: ArrayLike,
    *,
    threshold: float,
    window_samples: int,
    horizon_samples: int,
    stride_samples: int,
    gaussian: ArrayLike | None = None,
    fractions: tuple[float, float, float] = (0.60, 0.15, 0.25),
    definitions: FadeDefinitions = DEFAULT_DEFINITIONS,
) -> OutageDataset:
    """Assemble features, labels and a gapped temporal split in one call.

    The split gap is set to ``ceil((window_samples + horizon_samples) /
    stride_samples)`` rows, which is the smallest gap that makes the window and
    horizon of the last training row disjoint from those of the first
    calibration row. Anything smaller leaks.

    Parameters
    ----------
    amplitude:
        Amplitude record, strictly positive.
    threshold:
        Outage threshold, same units as ``amplitude``.
    window_samples, horizon_samples, stride_samples:
        Observation window, prediction horizon, spacing of decision instants,
        all in samples.
    gaussian:
        Optional AR(1) level series aligned with ``amplitude``, for the
        analytic predictor.
    fractions:
        Train / calibration / test fractions of the decision instants, in time
        order.
    definitions:
        Crossing definitions for the labels.

    Returns
    -------
    OutageDataset
    """
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    idx = decision_indices(
        a.size,
        window_samples=window_samples,
        horizon_samples=horizon_samples,
        stride_samples=stride_samples,
    )
    x = extract_features(
        a, index=idx, window_samples=window_samples, threshold=threshold
    )
    y = onset_labels(
        a,
        index=idx,
        threshold=threshold,
        horizon_samples=horizon_samples,
        definitions=definitions,
    )
    gap = int(np.ceil((int(window_samples) + int(horizon_samples)) / int(stride_samples)))
    split = temporal_split(y, fractions=fractions, gap_rows=gap)
    g: NDArray[np.float64] | None = None
    if gaussian is not None:
        garr = np.asarray(gaussian, dtype=np.float64).ravel()
        if garr.size != a.size:
            raise ValueError(
                f"gaussian has {garr.size} samples but amplitude has {a.size}; they must "
                "be the same record"
            )
        g = garr[idx]
    return OutageDataset(
        x=x,
        y=y,
        index=idx,
        gaussian=g,
        split=split,
        base_rate=float(y.mean()),
        configuration={
            "n_samples": int(a.size),
            "threshold": float(threshold),
            "window_samples": int(window_samples),
            "horizon_samples": int(horizon_samples),
            "stride_samples": int(stride_samples),
            "gap_rows": gap,
            "n_features": len(FEATURE_NAMES),
        },
    )
