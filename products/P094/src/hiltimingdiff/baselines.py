"""The three baselines of SPEC §4, in the order the specification fixes.

Baseline 1 — the declared timing model
--------------------------------------
``declared[s,c] = declared_ops(W,K)[s] * ns_per_op`` against
``hw_measured[s,c]``. This is the object being refuted, and it is implemented
at full strength on purpose: the operation count is *correct*, and it is wrong
as a timing proxy for reasons that are not arithmetic. An under-strength
version of this baseline would inflate the product's apparent finding, which is
a defect class this mission has shipped before.

Baseline 2 — one scalar, fitted on some stages, applied to others
-----------------------------------------------------------------
``k = median(hw_measured / sim_measured)`` over the TRAINING stages only, then
``predicted_hw[s] = k * sim_measured[s]`` on the HELD-OUT stages. The reported
figure is the mean relative error on the held-out stages. The question is the
one a team actually faces: they cannot put every stage on the board, so can a
factor measured on the stages they did measure be trusted on a stage they did
not?

The split is a choice and a choice can be gamed, so three things are reported
and not one:

* the **declared split**, pinned as a literal below, which is the
  vectorised/interpreted split SPEC §7 names and is therefore the *hardest*
  split for a single factor rather than a flattering one;
* **leave-one-stage-out** over all six stages, which involves no split choice;
* the **full distribution over all twenty three-and-three splits**, with its
  minimum, median and maximum, so that the declared split can be located inside
  it rather than taken on trust.

Baseline 3 — ground truth with an interval
------------------------------------------
``hw_measured[s]`` with a bootstrap interval **whose resampling unit is the
pass** (SPEC §4, non-negotiable). See :mod:`hiltimingdiff.bootstrap`.

Units: nanoseconds for every time, dimensionless for every ratio and relative
error.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bootstrap import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    RESAMPLES,
    RESAMPLING_UNIT,
    Interval,
    bootstrap_statistic,
)
from .divergence import PassUnits
from .simulate import SimulationResult

#: The declared training/held-out split for baseline 2, pinned as literals.
#: It is the vectorised/interpreted split SPEC §7 names as the mechanism, so it
#: is the split under which a single global factor has the least chance of
#: transferring. Choosing the hardest declared split rather than a kind one is
#: deliberate: a flattering split would make the secondary pre-registration
#: easier to confirm and the figure less useful.
TRAIN_STAGES: tuple[str, ...] = ("ingest", "background", "threshold")

#: The held-out stages of the declared split.
HOLDOUT_STAGES: tuple[str, ...] = ("centroid", "gate", "track")

#: SPEC §7's secondary pre-registration: baseline 2's mean relative transfer
#: error on held-out stages will EXCEED this value. Pinned as a literal so the
#: verdict cannot drift with the text.
PREREGISTERED_TRANSFER_ERROR = 0.5

#: SPEC §7's primary pre-registration: the divergence spread will EXCEED this.
PREREGISTERED_SPREAD = 3.0


@dataclass(frozen=True)
class Baseline1:
    """The declared timing model against hardware ground truth."""

    stages: tuple[str, ...]
    ns_per_op: float
    declared_ns: dict[str, float]
    hw_median_ns: dict[str, float]
    relative_error: dict[str, float]
    signed_relative_error: dict[str, float]
    mean_relative_error: float
    max_relative_error: float
    worst_stage: str
    total_declared_ns: float
    total_hw_ns: float
    total_relative_error: float

    def frame(self) -> pd.DataFrame:
        """Per-stage table in pipeline order."""
        return pd.DataFrame(
            [
                {
                    "stage": stage,
                    "declared_ns": self.declared_ns[stage],
                    "hw_median_ns": self.hw_median_ns[stage],
                    "signed_relative_error": self.signed_relative_error[stage],
                    "relative_error": self.relative_error[stage],
                }
                for stage in self.stages
            ]
        )


@dataclass(frozen=True)
class TransferSplit:
    """One training/held-out split of baseline 2."""

    train: tuple[str, ...]
    holdout: tuple[str, ...]
    k: float
    predicted_ns: dict[str, float]
    relative_error: dict[str, float]
    mean_relative_error: float
    max_relative_error: float


@dataclass(frozen=True)
class Baseline2:
    """Single-scalar transfer from measured stages to unmeasured ones."""

    stages: tuple[str, ...]
    hw_median_ns: dict[str, float]
    sim_median_ns: dict[str, float]
    per_stage_ratio: dict[str, float]
    declared: TransferSplit
    declared_interval: Interval
    leave_one_out: dict[str, float]
    leave_one_out_mean: float
    all_splits: tuple[TransferSplit, ...]
    split_errors: tuple[float, ...]
    k_global: float

    @property
    def split_error_min(self) -> float:
        """Smallest mean held-out relative error over all three-and-three splits."""
        return float(min(self.split_errors))

    @property
    def split_error_median(self) -> float:
        """Median mean held-out relative error over all splits."""
        return float(np.median(np.asarray(self.split_errors, dtype=np.float64)))

    @property
    def split_error_max(self) -> float:
        """Largest mean held-out relative error over all splits."""
        return float(max(self.split_errors))

    @property
    def splits_above_preregistration(self) -> int:
        """How many splits exceed SPEC §7's predicted 0.5."""
        return int(
            sum(1 for value in self.split_errors if value > PREREGISTERED_TRANSFER_ERROR)
        )


@dataclass(frozen=True)
class Baseline3:
    """Hardware ground truth with a pass-unit bootstrap interval per stage."""

    stages: tuple[str, ...]
    median_ns: dict[str, float]
    interval: dict[str, Interval]
    n_units: int
    unit: str = RESAMPLING_UNIT
    confidence: float = CONFIDENCE

    def frame(self) -> pd.DataFrame:
        """Per-stage median with interval bounds, in pipeline order."""
        return pd.DataFrame(
            [
                {
                    "stage": stage,
                    "median_ns": self.median_ns[stage],
                    "low_ns": self.interval[stage].low,
                    "high_ns": self.interval[stage].high,
                    "relative_width": self.interval[stage].relative_width,
                }
                for stage in self.stages
            ]
        )


def baseline_1(units: PassUnits, ns_per_op: float) -> Baseline1:
    """Baseline 1: the declared operation-count timing model.

    ``relative_error[s] = |declared[s] - hw[s]| / hw[s]`` on the per-stage
    medians over pass units. The signed form is kept beside it because the
    direction matters: a declared model that *under*-predicts a stage gives a
    schedulability analysis false headroom, which is the dangerous direction.
    """
    hw = np.median(units.hw, axis=0)
    ops = np.median(units.ops, axis=0)
    declared = ops * float(ns_per_op)
    signed = (declared - hw) / hw
    absolute = np.abs(signed)
    worst = int(np.argmax(absolute))
    total_declared = float(declared.sum())
    total_hw = float(hw.sum())
    return Baseline1(
        stages=units.stages,
        ns_per_op=float(ns_per_op),
        declared_ns=dict(zip(units.stages, (float(v) for v in declared), strict=True)),
        hw_median_ns=dict(zip(units.stages, (float(v) for v in hw), strict=True)),
        relative_error=dict(zip(units.stages, (float(v) for v in absolute), strict=True)),
        signed_relative_error=dict(
            zip(units.stages, (float(v) for v in signed), strict=True)
        ),
        mean_relative_error=float(absolute.mean()),
        max_relative_error=float(absolute.max()),
        worst_stage=units.stages[worst],
        total_declared_ns=total_declared,
        total_hw_ns=total_hw,
        total_relative_error=float(abs(total_declared - total_hw) / total_hw),
    )


def _paired_cell_ratios(
    units: PassUnits, simulation: SimulationResult
) -> dict[str, np.ndarray]:
    """Per-cell ``hw / sim`` ratio per stage, from the per-cell medians.

    One ratio per ``(stage, cell)``: the median over that cell's passes in
    hardware mode divided by the median over that cell's passes in simulation
    mode. Pairing by cell rather than pooling across cells keeps the frame
    dimension and the candidate load matched on both sides of the ratio.
    """
    hardware = pd.DataFrame(units.hw, columns=list(units.stages))
    hardware["cell"] = list(units.cells)
    hw_by_cell = hardware.groupby("cell").median(numeric_only=True)
    sim_by_cell = (
        simulation.table.pivot_table(
            index="cell", columns="stage", values="median_ns", aggfunc="median"
        )
        .reindex(columns=list(units.stages))
    )
    shared = [cell for cell in hw_by_cell.index if cell in set(sim_by_cell.index)]
    if not shared:
        raise ValueError(
            "no cell is present in both modes; hardware cells "
            f"{sorted(hw_by_cell.index)} and simulation cells {sorted(sim_by_cell.index)}"
        )
    ratios: dict[str, np.ndarray] = {}
    for stage in units.stages:
        top = hw_by_cell.loc[shared, stage].to_numpy(dtype=np.float64)
        bottom = sim_by_cell.loc[shared, stage].to_numpy(dtype=np.float64)
        ratios[stage] = top / bottom
    return ratios


def _split(
    train: tuple[str, ...],
    holdout: tuple[str, ...],
    ratios: dict[str, np.ndarray],
    hw_median: dict[str, float],
    sim_median: dict[str, float],
) -> TransferSplit:
    """Fit ``k`` on ``train`` and score it on ``holdout``."""
    pooled = np.concatenate([ratios[stage] for stage in train])
    k = float(np.median(pooled))
    predicted = {stage: k * sim_median[stage] for stage in holdout}
    errors = {
        stage: float(abs(predicted[stage] - hw_median[stage]) / hw_median[stage])
        for stage in holdout
    }
    values = np.asarray([errors[stage] for stage in holdout], dtype=np.float64)
    return TransferSplit(
        train=tuple(train),
        holdout=tuple(holdout),
        k=k,
        predicted_ns=predicted,
        relative_error=errors,
        mean_relative_error=float(values.mean()),
        max_relative_error=float(values.max()),
    )


def baseline_2(
    units: PassUnits,
    simulation: SimulationResult,
    *,
    train: tuple[str, ...] = TRAIN_STAGES,
    holdout: tuple[str, ...] = HOLDOUT_STAGES,
    confidence: float = CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Baseline2:
    """Baseline 2: one scalar fitted on training stages, scored on held-out ones.

    Raises
    ------
    ValueError
        If the split is not a partition of the campaign's stages, or if a stage
        is in both halves. A split that overlaps would score a factor on data it
        was fitted on, which is the one thing this baseline exists to avoid.
    """
    stages = units.stages
    unknown = [stage for stage in tuple(train) + tuple(holdout) if stage not in stages]
    if unknown:
        raise ValueError(
            f"unknown stage(s) in the split: {', '.join(unknown)}; "
            f"campaign stages are {', '.join(stages)}"
        )
    overlap = sorted(set(train) & set(holdout))
    if overlap:
        raise ValueError(
            f"training and held-out stages overlap on {', '.join(overlap)}: a factor "
            "scored on a stage it was fitted on is not a transfer measurement"
        )
    if not train or not holdout:
        raise ValueError("both halves of the split must be non-empty")

    ratios = _paired_cell_ratios(units, simulation)
    hw_median = {
        stage: float(np.median(units.hw[:, index]))
        for index, stage in enumerate(stages)
    }
    sim_median = simulation.stage_medians_ns()

    declared_split = _split(tuple(train), tuple(holdout), ratios, hw_median, sim_median)

    # Interval on the declared split's held-out error. The resampling unit is
    # the pass, as everywhere else in this product: the hardware medians the
    # error is measured against are per-pass quantities, so a pass resample is
    # what carries the hardware-side sampling variation into the error.
    stage_columns = {stage: index for index, stage in enumerate(stages)}
    holdout_columns = [stage_columns[stage] for stage in holdout]
    predicted = np.asarray(
        [declared_split.predicted_ns[stage] for stage in holdout], dtype=np.float64
    )
    holdout_hw = units.hw[:, holdout_columns]

    def holdout_error(sample: np.ndarray) -> float:
        medians = np.median(sample.reshape(-1, len(holdout_columns)), axis=0)
        return float(np.mean(np.abs(predicted - medians) / medians))

    flattened = holdout_hw.reshape(holdout_hw.shape[0], -1)
    interval = bootstrap_statistic(
        np.arange(flattened.shape[0], dtype=np.float64),
        lambda rows: holdout_error(flattened[rows.astype(np.int64)]),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )

    leave_one_out: dict[str, float] = {}
    for stage in stages:
        rest = tuple(other for other in stages if other != stage)
        leave_one_out[stage] = _split(
            rest, (stage,), ratios, hw_median, sim_median
        ).mean_relative_error
    loo_values = np.asarray([leave_one_out[stage] for stage in stages], dtype=np.float64)

    all_splits: list[TransferSplit] = []
    for combination in itertools.combinations(stages, len(stages) // 2):
        rest = tuple(stage for stage in stages if stage not in combination)
        all_splits.append(_split(combination, rest, ratios, hw_median, sim_median))

    return Baseline2(
        stages=stages,
        hw_median_ns=hw_median,
        sim_median_ns=sim_median,
        per_stage_ratio={stage: float(np.median(ratios[stage])) for stage in stages},
        declared=declared_split,
        declared_interval=interval,
        leave_one_out=leave_one_out,
        leave_one_out_mean=float(loo_values.mean()),
        all_splits=tuple(all_splits),
        split_errors=tuple(split.mean_relative_error for split in all_splits),
        k_global=float(np.median(np.concatenate([ratios[stage] for stage in stages]))),
    )


def baseline_3(
    units: PassUnits,
    *,
    confidence: float = CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Baseline3:
    """Baseline 3: hardware ground truth, with a pass-unit interval per stage.

    The resampling unit is the pass. It is recorded on every returned
    :class:`~hiltimingdiff.bootstrap.Interval` as ``unit`` and asserted by the
    test suite, because SPEC §4 declares it and an interval built on the wrong
    unit is narrower than the data supports rather than visibly wrong.
    """
    intervals: dict[str, Interval] = {}
    medians: dict[str, float] = {}
    for index, stage in enumerate(units.stages):
        column = units.hw[:, index]
        medians[stage] = float(np.median(column))
        intervals[stage] = bootstrap_statistic(
            column,
            lambda sample: float(np.median(sample)),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
    return Baseline3(
        stages=units.stages,
        median_ns=medians,
        interval=intervals,
        n_units=units.n_units,
        confidence=confidence,
    )
