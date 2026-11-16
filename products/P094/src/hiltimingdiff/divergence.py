"""The five headline outputs of SPEC §5, with pass-unit intervals.

Definitions, exactly as SPEC §5 states them
-------------------------------------------

``R[s] = median(hw_measured[s]) / median(declared[s])``
    A ratio of medians, not a median of ratios. The two are different
    statistics and the specification names the first.

``spread = max(R) / min(R)``
    The number SPEC §5 calls the one that matters. At 1.0 the declared model is
    uniformly wrong and a schedulability ordering built on it survives; far
    from 1.0 the model is stage-dependently wrong and the ordering does not.

``rank disagreement``
    Kendall's tau-b between the stage ordering by ``declared`` and by
    ``hw_measured``, plus the explicit permutation that carries one ordering to
    the other.

``argmax(R)``, ``argmin(R)``
    The least and most trustworthy stages as schedulability inputs, named.

One invariance worth stating, because it decides which figures are exactly
reproducible
------------------------------------------------------------------------------

``declared[s,c] = declared_ops(W,K)[s] * ns_per_op`` with a single global
``ns_per_op`` calibrated on the simulation host. The spread, the Kendall
statistic, the permutation and both arg-extrema are therefore **invariant**
under ``ns_per_op``: it cancels in the spread and is a positive monotone
rescaling of every ``R[s]``, which no rank statistic can see. Only the
*levels* of ``R[s]`` carry the host calibration.

So this module reports two forms and labels both:

* ``ns_per_declared_op[s] = median(hw_measured[s]) / median(declared_ops[s])``,
  in nanoseconds per declared operation, which depends on nothing but the
  committed traces and is bit-for-bit reproducible from them;
* ``R[s] = ns_per_declared_op[s] / ns_per_op``, which is the specification's
  ratio and moves with the host calibration.

Units: ``median_ns`` and every hardware quantity are nanoseconds;
``declared_ops`` is dimensionless; ``ns_per_declared_op`` is ns per declared
operation; ``R`` and ``spread`` are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bootstrap import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    RESAMPLES,
    RESAMPLING_UNIT,
    Interval,
    percentile_interval,
    unit_resamples,
)
from .traces import Campaign, hardware_pass_table


@dataclass(frozen=True)
class PassUnits:
    """Per-pass hardware medians and declared operation counts, aligned.

    ``hw[u, s]`` is the median stage time in nanoseconds of stage ``s`` over
    every timed frame of pass unit ``u``; ``ops[u, s]`` is that pass's declared
    operation count for the same stage, evaluated at the pass's own ``k_mean``.
    One unit is one ``(cell, pass_idx)`` pair: the declared resampling unit.
    """

    stages: tuple[str, ...]
    cells: tuple[str, ...]
    pass_index: np.ndarray
    hw: np.ndarray
    ops: np.ndarray

    def __post_init__(self) -> None:
        if self.hw.shape != self.ops.shape:
            raise ValueError(
                f"hw {self.hw.shape} and ops {self.ops.shape} must have the same shape"
            )
        if self.hw.shape[1] != len(self.stages):
            raise ValueError(
                f"hw has {self.hw.shape[1]} stage columns but {len(self.stages)} "
                "stage names were given"
            )

    @property
    def n_units(self) -> int:
        """Number of pass units, the bootstrap's sample size."""
        return int(self.hw.shape[0])

    @property
    def n_stages(self) -> int:
        """Number of stages."""
        return int(self.hw.shape[1])

    def stage_column(self, stage: str) -> int:
        """Column index of ``stage``."""
        if stage not in self.stages:
            raise KeyError(f"{stage!r} is not one of {', '.join(self.stages)}")
        return self.stages.index(stage)


def pass_units(campaign: Campaign) -> PassUnits:
    """Build the per-pass unit table from a loaded campaign.

    The unit is the ``(cell, pass_idx)`` pair and every stage of one pass is
    one row, so a bootstrap resample of rows carries a pass's whole 6-vector.
    """
    table = hardware_pass_table(campaign)
    stages = campaign.stage_names
    wide_hw = table.pivot_table(
        index=["cell", "pass_idx"], columns="stage", values="median_ns", aggfunc="first"
    )
    wide_ops = table.pivot_table(
        index=["cell", "pass_idx"], columns="stage", values="declared_ops", aggfunc="first"
    )
    wide_hw = wide_hw.reindex(columns=list(stages))
    wide_ops = wide_ops.reindex(columns=list(stages))
    if wide_hw.isna().to_numpy().any() or wide_ops.isna().to_numpy().any():
        raise ValueError(
            "the per-pass table has a hole: some (cell, pass) unit is missing a "
            "stage. A unit that does not carry all six stages cannot be a joint "
            "resampling unit, and no interval is reported from a holed table."
        )
    index = wide_hw.index
    return PassUnits(
        stages=tuple(stages),
        cells=tuple(str(value) for value in index.get_level_values("cell")),
        pass_index=np.asarray(index.get_level_values("pass_idx"), dtype=np.int64),
        hw=wide_hw.to_numpy(dtype=np.float64),
        ops=wide_ops.to_numpy(dtype=np.float64),
    )


def kendall_tau_b(first: np.ndarray, second: np.ndarray) -> float:
    """Kendall's tau-b between two rankings of the same items.

    tau-b is the concordance statistic corrected for ties on either side:

    ``tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))``

    with ``n0 = n(n-1)/2``, ``C`` concordant and ``D`` discordant pairs, ``n1``
    the tied pairs in ``first`` and ``n2`` the tied pairs in ``second``
    (Kendall 1945, *Biometrika* 33, 239-251). Written out here rather than
    taken from a library so that the statistic this product publishes is one
    whose definition is visible; ``validation/validate_independent_bound.py``
    cross-checks it against ``scipy.stats.kendalltau``, which is a different
    implementation of the same definition.

    Returns ``nan`` when either side is constant, because tau-b is then
    undefined rather than zero.

    Concordance is decided on the PRODUCT OF SIGNS of the two differences and
    not on the product of the differences themselves. A property test found the
    reason: at ``values = [0.0, 1.3637850869171467e-163]`` the difference is
    representable but its square underflows to exactly ``0.0`` in float64, so a
    product-based test reads a concordant pair as a tied one and returns
    ``0.0`` where the answer is ``1.0``. The sign product cannot underflow. The
    counterexample is pinned in ``tests/test_properties.py`` as
    ``test_regression_tau_b_survives_a_difference_whose_square_underflows``.
    """
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError(f"rankings must be the same length: {left.shape} vs {right.shape}")
    if left.size < 2:
        raise ValueError("Kendall's tau-b needs at least 2 items")
    concordant = 0
    discordant = 0
    tied_left = 0
    tied_right = 0
    for i in range(left.size - 1):
        delta_left = left[i + 1 :] - left[i]
        delta_right = right[i + 1 :] - right[i]
        product = np.sign(delta_left) * np.sign(delta_right)
        concordant += int(np.count_nonzero(product > 0.0))
        discordant += int(np.count_nonzero(product < 0.0))
        tied_left += int(np.count_nonzero(delta_left == 0.0))
        tied_right += int(np.count_nonzero(delta_right == 0.0))
    n_pairs = left.size * (left.size - 1) // 2
    scale = np.sqrt(float(n_pairs - tied_left) * float(n_pairs - tied_right))
    if scale == 0.0:
        return float("nan")
    return float(concordant - discordant) / float(scale)


def ordering(values: np.ndarray) -> np.ndarray:
    """Indices that sort ``values`` ascending, ties broken by index.

    Ascending, so position 0 is the cheapest stage. Deterministic under ties
    because ``kind="stable"`` keeps pipeline order, which matters for a
    published permutation.
    """
    return np.argsort(np.asarray(values, dtype=np.float64), kind="stable")


def rank_permutation(declared: np.ndarray, measured: np.ndarray) -> tuple[int, ...]:
    """The explicit permutation carrying the declared order to the measured one.

    Entry ``i`` is the position, in the declared ordering, of the stage that
    sits at position ``i`` of the measured ordering. The identity permutation
    means the two orderings agree.
    """
    declared_order = ordering(declared)
    measured_order = ordering(measured)
    position_in_declared = {int(stage): rank for rank, stage in enumerate(declared_order)}
    return tuple(position_in_declared[int(stage)] for stage in measured_order)


@dataclass(frozen=True)
class Divergence:
    """The five headline outputs of SPEC §5, each with its pass-unit interval."""

    stages: tuple[str, ...]
    ns_per_op: float
    hw_median_ns: dict[str, float]
    declared_ops_median: dict[str, float]
    declared_ns: dict[str, float]
    ns_per_declared_op: dict[str, float]
    ratio: dict[str, float]
    ratio_interval: dict[str, Interval]
    spread: float
    spread_interval: Interval
    tau_b: float
    tau_b_interval: Interval
    permutation: tuple[int, ...]
    declared_order: tuple[str, ...]
    measured_order: tuple[str, ...]
    argmax: str
    argmin: str
    argmax_share: float
    argmin_share: float
    n_units: int
    unit: str = RESAMPLING_UNIT
    confidence: float = CONFIDENCE
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_rows(self) -> list[dict[str, object]]:
        """One dict per stage, for tabular rendering."""
        return [
            {
                "stage": stage,
                "hw_median_ns": self.hw_median_ns[stage],
                "declared_ops": self.declared_ops_median[stage],
                "declared_ns": self.declared_ns[stage],
                "ns_per_declared_op": self.ns_per_declared_op[stage],
                "R": self.ratio[stage],
                "R_low": self.ratio_interval[stage].low,
                "R_high": self.ratio_interval[stage].high,
            }
            for stage in self.stages
        ]

    def frame(self) -> pd.DataFrame:
        """The per-stage table as a DataFrame in pipeline order."""
        return pd.DataFrame(self.as_rows())


def divergence(
    units: PassUnits,
    ns_per_op: float,
    *,
    confidence: float = CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Divergence:
    """Compute every SPEC §5 output with a joint pass-unit bootstrap.

    Parameters
    ----------
    units:
        Per-pass hardware medians and declared operation counts.
    ns_per_op:
        The single global scalar of SPEC §3, in nanoseconds per declared
        operation, calibrated once on the simulation host. Only the levels of
        ``R`` depend on it; the spread, tau-b, the permutation and both
        arg-extrema do not.
    """
    if not np.isfinite(ns_per_op) or ns_per_op <= 0.0:
        raise ValueError(
            f"ns_per_op must be a positive finite nanosecond cost, got {ns_per_op!r}"
        )
    hw_median = np.median(units.hw, axis=0)
    ops_median = np.median(units.ops, axis=0)
    if np.any(ops_median <= 0.0):
        raise ValueError(
            "a stage has a non-positive median declared operation count; "
            f"medians were {ops_median.tolist()}"
        )
    declared_ns = ops_median * float(ns_per_op)
    per_op = hw_median / ops_median
    ratio = hw_median / declared_ns

    indices = unit_resamples(units.n_units, resamples=resamples, seed=seed)
    ratio_replicates = np.empty((indices.shape[0], units.n_stages), dtype=np.float64)
    spread_replicates = np.empty(indices.shape[0], dtype=np.float64)
    tau_replicates = np.empty(indices.shape[0], dtype=np.float64)
    for row_index, row in enumerate(indices):
        hw_sample = np.median(units.hw[row], axis=0)
        ops_sample = np.median(units.ops[row], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            replicate = hw_sample / (ops_sample * float(ns_per_op))
        ratio_replicates[row_index] = replicate
        finite = replicate[np.isfinite(replicate) & (replicate > 0.0)]
        spread_replicates[row_index] = (
            float(finite.max() / finite.min()) if finite.size == units.n_stages else np.nan
        )
        tau_replicates[row_index] = kendall_tau_b(ops_sample, hw_sample)

    ratio_interval = {
        units.stages[column]: percentile_interval(
            float(ratio[column]),
            ratio_replicates[:, column],
            confidence=confidence,
            units=units.n_units,
        )
        for column in range(units.n_stages)
    }
    spread = float(ratio.max() / ratio.min())
    spread_interval = percentile_interval(
        spread, spread_replicates, confidence=confidence, units=units.n_units
    )
    tau = kendall_tau_b(ops_median, hw_median)
    tau_interval = percentile_interval(
        tau, tau_replicates, confidence=confidence, units=units.n_units
    )

    declared_order = ordering(ops_median)
    measured_order = ordering(hw_median)
    total_hw = float(hw_median.sum())
    argmax_column = int(np.argmax(ratio))
    argmin_column = int(np.argmin(ratio))
    return Divergence(
        stages=units.stages,
        ns_per_op=float(ns_per_op),
        hw_median_ns=dict(zip(units.stages, (float(v) for v in hw_median), strict=True)),
        declared_ops_median=dict(
            zip(units.stages, (float(v) for v in ops_median), strict=True)
        ),
        declared_ns=dict(zip(units.stages, (float(v) for v in declared_ns), strict=True)),
        ns_per_declared_op=dict(zip(units.stages, (float(v) for v in per_op), strict=True)),
        ratio=dict(zip(units.stages, (float(v) for v in ratio), strict=True)),
        ratio_interval=ratio_interval,
        spread=spread,
        spread_interval=spread_interval,
        tau_b=float(tau),
        tau_b_interval=tau_interval,
        permutation=rank_permutation(ops_median, hw_median),
        declared_order=tuple(units.stages[int(i)] for i in declared_order),
        measured_order=tuple(units.stages[int(i)] for i in measured_order),
        argmax=units.stages[argmax_column],
        argmin=units.stages[argmin_column],
        argmax_share=float(hw_median[argmax_column] / total_hw),
        argmin_share=float(hw_median[argmin_column] / total_hw),
        n_units=units.n_units,
        confidence=confidence,
    )
