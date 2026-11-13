"""Windowing and descriptor extraction for the learned predictor's dataset.

The campaign is split into fixed-length windows. Each window contributes

- **descriptors** an operator could observe WITHOUT running the task: the
  declared CPU- and memory-stressor counts, and the monitor's measured foreign
  CPU occupancy, our own and the stressors' occupancy, load average, mean and
  minimum core clock, and die temperature, each averaged over the window's 1 Hz
  samples;
- a **target**: the window's own counted miss rate, by the measurement model in
  :mod:`deadlinerisk.trace` -- misses and releases are both summed over the
  executed rows whose RELEASE falls inside the window, so a dropped release is
  attributed to the window of the release that was dropped.

Window length is a declared constant. 10 s gives 60 windows from the 600 s
ground-truth run and 15 from each 150 s mix, which is the largest sample the
campaign supports at a length long enough for a window's counted rate to be
something other than zero or one.

A window with fewer than :data:`MIN_RELEASES_PER_WINDOW` releases or with no
monitor samples is dropped, and the count dropped is reported by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import Campaign

#: Window length, seconds.
WINDOW_S = 10.0

#: A window with fewer releases than this is dropped as unrepresentative.
MIN_RELEASES_PER_WINDOW = 100

#: Descriptor names, in column order. Every one is observable without running
#: the task under analysis.
FEATURE_NAMES: tuple[str, ...] = (
    "n_cpu_stress",
    "n_mem_stress",
    "n_workers",
    "foreign_pct",
    "task_pct",
    "stress_pct",
    "load1",
    "clk_mean_khz",
    "clk_min_khz",
    "tj_degC",
)


@dataclass(frozen=True)
class WindowSet:
    """A design matrix, targets, run labels and window start times.

    Attributes
    ----------
    features
        ``(n_windows, n_features)`` float array, columns in
        :data:`FEATURE_NAMES` order.
    targets
        ``(n_windows,)`` counted miss rate per window, dimensionless.
    runs
        ``(n_windows,)`` run label per window, used as the grouping variable
        for the held-out-mix split.
    window_start_s
        ``(n_windows,)`` window start time within its run, seconds.
    releases
        ``(n_windows,)`` release count per window.
    dropped_windows
        Windows discarded, and why, as ``(run, start_s, reason)`` triples.
    """

    features: np.ndarray
    targets: np.ndarray
    runs: np.ndarray
    window_start_s: np.ndarray
    releases: np.ndarray
    feature_names: tuple[str, ...]
    window_s: float
    dropped_windows: tuple[tuple[str, float, str], ...]

    def __len__(self) -> int:
        return int(self.targets.size)


def extract_windows(
    campaign: Campaign,
    window_s: float = WINDOW_S,
    min_releases: int = MIN_RELEASES_PER_WINDOW,
) -> WindowSet:
    """Window every run of ``campaign`` and extract descriptors and targets.

    Raises
    ------
    ValueError
        On a non-positive window length, or if no window survives, which means
        the campaign is too short or its monitor files are absent.
    """
    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError(f"window_s must be finite and > 0, got {window_s!r}")
    rows: list[list[float]] = []
    targets: list[float] = []
    runs: list[str] = []
    starts: list[float] = []
    releases: list[int] = []
    dropped: list[tuple[str, float, str]] = []

    for run in sorted(campaign.jobs):
        jobs = campaign.jobs[run]
        monitor = campaign.monitors.get(run)
        try:
            n_cpu, n_mem = campaign.stressors(run)
        except KeyError:
            dropped.append((run, -1.0, "no stressor counts in the manifest"))
            continue
        if monitor is None or len(monitor) == 0:
            dropped.append((run, -1.0, "no monitor samples"))
            continue
        if len(jobs) == 0:
            dropped.append((run, -1.0, "no executed job rows"))
            continue
        span = float(jobs.release_s[-1])
        n_windows = int(np.floor(span / window_s))
        for index in range(n_windows):
            lo = index * window_s
            hi = lo + window_s
            mask = (jobs.release_s >= lo) & (jobs.release_s < hi)
            n_rows = int(np.count_nonzero(mask))
            if n_rows == 0:
                dropped.append((run, lo, "no executed rows in the window"))
                continue
            drops = int(jobs.dropped_before[mask].sum())
            window_releases = n_rows + drops
            window_misses = int(jobs.missed[mask].sum()) + drops
            if window_releases < min_releases:
                dropped.append(
                    (run, lo, f"{window_releases} releases, fewer than {min_releases}")
                )
                continue
            m_mask = (monitor.t_s >= lo) & (monitor.t_s < hi)
            if not np.any(m_mask):
                dropped.append((run, lo, "no monitor samples in the window"))
                continue
            rows.append(
                [
                    float(n_cpu),
                    float(n_mem),
                    float(n_cpu + n_mem),
                    float(monitor.foreign_pct[m_mask].mean()),
                    float(monitor.task_pct[m_mask].mean()),
                    float(monitor.stress_pct[m_mask].mean()),
                    float(monitor.load1[m_mask].mean()),
                    float(monitor.clk_mean_khz[m_mask].mean()),
                    float(monitor.clk_min_khz[m_mask].mean()),
                    float(monitor.tj_degC[m_mask].mean()),
                ]
            )
            targets.append(window_misses / window_releases)
            runs.append(run)
            starts.append(lo)
            releases.append(window_releases)

    if not rows:
        raise ValueError(
            "no window survived extraction. Reasons recorded: "
            + "; ".join(f"{run}@{start:.0f}s: {reason}" for run, start, reason in dropped[:5])
        )
    return WindowSet(
        features=np.asarray(rows, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        runs=np.asarray(runs, dtype=object),
        window_start_s=np.asarray(starts, dtype=np.float64),
        releases=np.asarray(releases, dtype=np.int64),
        feature_names=FEATURE_NAMES,
        window_s=float(window_s),
        dropped_windows=tuple(dropped),
    )


def grouped_splits(windows: WindowSet) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-run-out splits: ``(held_out_run, train_index, test_index)``.

    No window of a held-out run is ever in training, which is what "held-out
    workload mix" has to mean for the score to be about generalising to an
    unseen mix rather than to an unseen second of a seen mix.
    """
    labels = sorted(set(windows.runs.tolist()))
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label in labels:
        test = np.nonzero(windows.runs == label)[0]
        train = np.nonzero(windows.runs != label)[0]
        if train.size == 0 or test.size == 0:
            continue
        out.append((label, train, test))
    return out


def time_split(
    windows: WindowSet, holdout_fraction: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out TIME: the last ``holdout_fraction`` of every run's windows.

    Training sees only the earlier portion of each run, so a model cannot
    interpolate between windows that straddle a test window in time. Returns
    ``(train_index, test_index)``.

    Raises
    ------
    ValueError
        On a fraction outside ``(0, 1)``, or if either side comes out empty.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction!r}")
    train: list[int] = []
    test: list[int] = []
    for label in sorted(set(windows.runs.tolist())):
        index = np.nonzero(windows.runs == label)[0]
        order = index[np.argsort(windows.window_start_s[index])]
        cut = int(np.floor(order.size * (1.0 - holdout_fraction)))
        cut = max(1, min(order.size - 1, cut)) if order.size > 1 else order.size
        train.extend(order[:cut].tolist())
        test.extend(order[cut:].tolist())
    if not train or not test:
        raise ValueError(
            f"time_split produced {len(train)} training and {len(test)} test windows; "
            "the campaign has too few windows per run for this split"
        )
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(test), dtype=np.int64)
