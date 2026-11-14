"""Windowing and the LICENSED descriptor set for the learned predictor.

The dataset
-----------

Each pipeline cell is cut into fixed-length windows of :data:`WINDOW_S`
seconds, four cycles each. A window contributes

- **descriptors** that an operator could state from the configuration and read
  from the platform's own environment **without running the pipeline**, and
- a **target**: the window's mean ACTIVE rail power, mW, computed as the
  trapezoidal energy over that window's active windows divided by their
  measured duration.

The background floor cell contributes nothing: it ran no pipeline, so it has no
active windows and no target. That is stated here rather than handled by a
silent skip.

The licensed descriptors
------------------------

:data:`FEATURE_NAMES`, in column order:

====================== ===================================================
descriptor             where an operator gets it
====================== ===================================================
``duty``               the configuration
``cycle_s``            the configuration
``frame_dim``          the configuration
``tau``                the configuration, via the declared calibration rule
``cores_available``    the platform
``env_tj_c``           the platform's own thermal zone at the window start
``env_load1``          the platform's own load average at the window start
``env_clk_mean_khz``   the platform's own core clocks at the window start
``env_foreign_busy``   /proc/stat occupancy in pipeline-free monitor samples
``env_foreign_imputed`` 1 when the line above could not be separated
====================== ===================================================

``cycle_s`` and ``cores_available`` are constant across this campaign. They are
kept in the matrix because they are part of the licensed set and a reader
should see that they were offered to the model and carried no information,
rather than find them quietly absent.

**``env_load1`` is not pipeline-free.** The 1-minute load average includes the
pipeline's own run queue contribution. The specification names it as a licensed
descriptor and it is used, but the contamination is real and is stated in
``MODEL_CARD.md`` and measured by ``validation/validate_descriptor_leak.py``,
which refits without it and reports the cost.

**``env_foreign_busy`` is separable only at low duty.** The monitor samples
``/proc/stat`` once a second and each sample covers the preceding second. A
sample is pipeline-free only when no inference interval overlaps that second.
With a 2.0 s cycle, the sleep window is ``2*(1-duty)`` seconds long, so a
pipeline-free second can exist at duty 0.10 and 0.25 and cannot exist at duty
0.75 or 1.00. Where a window has no pipeline-free sample the descriptor is
imputed from the floor cell's own mean occupancy and ``env_foreign_imputed`` is
set to 1. The imputation is a declared constant from the same campaign, it is
flagged in the design matrix, and it is reported in ``MODEL_CARD.md`` as an
imputation rather than presented as a per-window measurement.

What is excluded, and enforced
------------------------------

Everything the pipeline itself produces: its measured power, its measured
latency, its CPU occupancy, its realised inference count and its realised
acceptance rate. The enforcement lives in
:func:`powerenvelope.predictor.assert_descriptors_licensed`, which is called by
the predictor's ``fit`` and raises
:class:`~powerenvelope.predictor.DescriptorLeakError`. It is a named check and
``tests/test_predictor.py`` asserts that it fires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .energy import active_windows, energy_over_windows, merge_intervals
from .schema import N_CORES
from .trace import FLOOR_CELL, Campaign, Cell

#: Window length, seconds: four cycles of the campaign's 2.0 s cycle. 76 s runs
#: give nine whole windows each, and 19 pipeline cells give 171 windows. A
#: shorter window would give more rows whose active duration is a handful of
#: samples; a longer one would give fewer than one hundred rows in total.
WINDOW_S = 8.0

#: A window with less than this much measured active duration is dropped. Four
#: sampler periods: below that, the window's mean active power is two or three
#: hardware-averaged readings and is not a mean of anything.
MIN_ACTIVE_S = 0.2

#: Descriptor names, in column order. Every one is observable without running
#: the pipeline under analysis. See this module's docstring for the caveats on
#: ``env_load1`` and ``env_foreign_busy``, which are stated and not hidden.
FEATURE_NAMES: tuple[str, ...] = (
    "duty",
    "cycle_s",
    "frame_dim",
    "tau",
    "cores_available",
    "env_tj_c",
    "env_load1",
    "env_clk_mean_khz",
    "env_foreign_busy",
    "env_foreign_imputed",
)


@dataclass(frozen=True)
class WindowSet:
    """A design matrix, targets and the grouping labels the splits use.

    Attributes
    ----------
    features
        ``(n_windows, len(FEATURE_NAMES))`` design matrix.
    targets
        ``(n_windows,)`` mean ACTIVE rail power per window, mW.
    cells
        Cell label per window.
    duties, frame_dims
        Grouping labels per window.
    t_start
        Window start on ``CLOCK_MONOTONIC``, seconds.
    active_s
        Measured active duration inside each window, seconds.
    dropped
        ``(cell, reason)`` for every window not included.
    foreign_imputed
        How many windows had their foreign-occupancy descriptor imputed.
    floor_foreign_busy
        The constant used for that imputation, measured on the floor cell.
    """

    features: np.ndarray
    targets: np.ndarray
    cells: np.ndarray
    duties: np.ndarray
    frame_dims: np.ndarray
    t_start: np.ndarray
    active_s: np.ndarray
    feature_names: tuple[str, ...]
    window_s: float
    dropped: tuple[tuple[str, str], ...]
    foreign_imputed: int
    floor_foreign_busy: float

    def __len__(self) -> int:
        return int(self.targets.size)

    def column(self, name: str) -> np.ndarray:
        """One descriptor column by name."""
        try:
            index = self.feature_names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a descriptor; present: {list(self.feature_names)}"
            ) from None
        return self.features[:, index]

    def without(self, drop: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
        """``(matrix, names)`` with the named descriptors removed."""
        keep = [index for index, name in enumerate(self.feature_names) if name not in drop]
        if not keep:
            raise ValueError("cannot drop every descriptor")
        return self.features[:, keep], tuple(self.feature_names[index] for index in keep)


def pipeline_free_monitor_mask(cell: Cell) -> np.ndarray:
    """Which monitor samples cover a second with no inference in it.

    Each monitor sample at ``t`` reports the ``/proc/stat`` occupancy of
    ``(t - 1, t]``. A sample is pipeline-free when no merged active window
    overlaps that interval, which is the only way the measured occupancy can be
    attributed to something other than this pipeline.

    Returns
    -------
    ndarray of bool
        One entry per monitor sample. All False for a cell whose duty leaves no
        whole pipeline-free second, which is the normal case at duty 0.75 and
        1.00.
    """
    if cell.decisions is None:
        return np.ones(len(cell.monitor), dtype=bool)
    starts, ends = active_windows(cell.decisions)
    lo = cell.monitor.t - cell.monitor.sample_interval_s
    hi = cell.monitor.t
    free = np.ones(lo.size, dtype=bool)
    for a, b in zip(starts, ends, strict=True):
        free &= ~((lo < b) & (hi > a))
    return free


def _environment_at(cell: Cell, t: float, floor_busy: float) -> tuple[np.ndarray, bool]:
    """Window-start environment descriptors, and whether occupancy was imputed."""
    monitor = cell.monitor
    at_or_before = np.nonzero(monitor.t <= t)[0]
    index = int(at_or_before[-1]) if at_or_before.size else 0
    clk_mean = monitor.clk_mean_khz[index]
    free = pipeline_free_monitor_mask(cell)
    inside = (monitor.t >= t) & (monitor.t < t + WINDOW_S) & free
    if np.any(inside):
        foreign = float(np.mean(monitor.cpu_busy_fraction[inside]))
        imputed = False
    else:
        foreign = floor_busy
        imputed = True
    return (
        np.asarray(
            [
                float(monitor.tj_c[index]),
                float(monitor.load1[index]),
                float(clk_mean),
                foreign,
                1.0 if imputed else 0.0,
            ],
            dtype=np.float64,
        ),
        imputed,
    )


def extract_windows(campaign: Campaign, window_s: float = WINDOW_S) -> WindowSet:
    """Cut every pipeline cell into windows and build the design matrix.

    Raises
    ------
    ValueError
        On a non-positive window length, or if no window survives.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    floor_busy = float("nan")
    if FLOOR_CELL in campaign.cells:
        floor_busy = float(np.mean(campaign.cells[FLOOR_CELL].monitor.cpu_busy_fraction))
    rows: list[np.ndarray] = []
    targets: list[float] = []
    labels: list[str] = []
    duties: list[float] = []
    dims: list[int] = []
    starts_out: list[float] = []
    active_out: list[float] = []
    dropped: list[tuple[str, str]] = []
    imputed_count = 0
    for cell in campaign.cells.values():
        if not cell.has_pipeline:
            dropped.append((cell.label, "no pipeline: the background floor cell has no target"))
            continue
        decisions = cell.require_decisions()
        a_lo, a_hi = active_windows(decisions)
        t0 = float(cell.power.t[0])
        t1 = float(cell.power.t[-1])
        n_windows = int(np.floor((t1 - t0) / window_s))
        if n_windows < 1:
            dropped.append((cell.label, "run shorter than one window"))
            continue
        for index in range(n_windows):
            lo = t0 + index * window_s
            hi = lo + window_s
            clipped_lo = np.clip(a_lo, lo, hi)
            clipped_hi = np.clip(a_hi, lo, hi)
            keep = clipped_hi > clipped_lo
            if not np.any(keep):
                dropped.append((cell.label, f"window {index}: no active interval"))
                continue
            merged_lo, merged_hi = merge_intervals(clipped_lo[keep], clipped_hi[keep])
            window = energy_over_windows(cell.power, merged_lo, merged_hi)
            if not np.isfinite(window.mean_power_mw) or window.duration_s < MIN_ACTIVE_S:
                dropped.append(
                    (cell.label, f"window {index}: active duration {window.duration_s:.3f} s")
                )
                continue
            environment, imputed = _environment_at(cell, lo, floor_busy)
            if imputed:
                imputed_count += 1
            if not np.all(np.isfinite(environment)):
                dropped.append((cell.label, f"window {index}: non-finite environment"))
                continue
            rows.append(
                np.concatenate(
                    (
                        np.asarray(
                            [
                                cell.duty,
                                campaign.cycle_s,
                                float(cell.frame_dim),
                                cell.tau,
                                float(N_CORES),
                            ],
                            dtype=np.float64,
                        ),
                        environment,
                    )
                )
            )
            targets.append(window.mean_power_mw)
            labels.append(cell.label)
            duties.append(cell.duty)
            dims.append(cell.frame_dim)
            starts_out.append(lo)
            active_out.append(window.duration_s)
    if not rows:
        raise ValueError(
            "no window survived extraction. Reasons recorded per window: "
            f"{dropped[:6]}{' ...' if len(dropped) > 6 else ''}"
        )
    return WindowSet(
        features=np.vstack(rows),
        targets=np.asarray(targets, dtype=np.float64),
        cells=np.asarray(labels, dtype=object),
        duties=np.asarray(duties, dtype=np.float64),
        frame_dims=np.asarray(dims, dtype=np.int64),
        t_start=np.asarray(starts_out, dtype=np.float64),
        active_s=np.asarray(active_out, dtype=np.float64),
        feature_names=FEATURE_NAMES,
        window_s=window_s,
        dropped=tuple(dropped),
        foreign_imputed=imputed_count,
        floor_foreign_busy=floor_busy,
    )


def grouped_splits(
    windows: WindowSet, by: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-group-out splits, grouped so no test window is in training.

    Parameters
    ----------
    windows
        The dataset.
    by
        ``"duty"`` for held-out duty cycles or ``"frame_dim"`` for held-out
        workloads. The two are scored separately, never pooled: a model that
        interpolates across duty at a fixed frame dimension is answering a
        different question from one that extrapolates to an unseen workload.

    Returns
    -------
    list of tuple
        ``(label, train_index, test_index)`` per group.

    Raises
    ------
    ValueError
        On an unknown grouping, or when a grouping has fewer than two groups so
        that leave-one-out leaves no training data.
    """
    if by == "duty":
        values = windows.duties
        render = "{:.2f}".format
    elif by == "frame_dim":
        values = windows.frame_dims
        render = "{:.0f}".format
    else:
        raise ValueError(f"by must be 'duty' or 'frame_dim', got {by!r}")
    groups = np.unique(values)
    if groups.size < 2:
        raise ValueError(
            f"grouping by {by} gives {groups.size} group(s); leave-one-out needs at "
            "least two, or there is nothing to train on."
        )
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    for group in groups:
        test = np.nonzero(values == group)[0]
        train = np.nonzero(values != group)[0]
        splits.append((f"{by}={render(float(group))}", train, test))
    return splits
