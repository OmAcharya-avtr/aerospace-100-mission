"""The ordered study: baselines first, then the learned model, then the scores.

This module is the product's spine. It runs the four model families on one
record in the order the build guide requires -- the analytic baselines are
fitted and the empirical reference is bootstrapped before the learned model is
touched -- and returns everything any report needs. It makes no claim and prints
nothing.

**(39) The four correlation models compared.**

+------------------+----------------------------------------------------------+
| ``independent``  | ``rho = 0``. Baseline 1, equation (14). The thing being   |
|                  | refuted.                                                 |
| ``exponential``  | ``rho_0 exp(-d/L)``, pooled. Baseline 2 as specified,    |
|                  | equation (23).                                           |
| ``per_regime``   | the same, fitted per regime. The strengthened baseline,  |
|                  | equation (24).                                           |
| ``learned``      | the quantile tree ensemble, equations (35) to (38).      |
+------------------+----------------------------------------------------------+

Baseline 3, the empirical joint of equation (20), is not in the table: it is
the reference the table is scored against.

**(40) The scoring metric.** Mean absolute error in JOINT UNAVAILABILITY against
equation (20), over the rows of a split cell. Absolute and not squared, because
the quantity is a probability bounded below by zero and a squared error is
dominated by the few near pairs with the largest joint rates, which are also the
ones every model gets roughly right. The mean absolute error in the implied
CORRELATION is reported beside it, because the two can disagree: a model can be
closer in correlation and further in probability for a pair whose marginals are
extreme.

**(41) The comparison that decides the headline.** For each hold-out cell, the
margin

    margin = MAE(strongest baseline) - MAE(learned)

positive when the learned model is better. The margin is compared against the
paired bootstrap interval of equation (22) on the SAME margin, formed by
resampling the record and recomputing both errors against the resampled
reference. A margin whose interval contains zero is not evidence that the two
models differ. This is reported for both hold-out cells separately and is never
merged into one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .climatology import Climatology, estimate_climatology
from .copula import RHO_MAX, RHO_MIN, EpochNodes, epoch_nodes
from .correlation import (
    ExponentialModel,
    PerRegimeModel,
    fit_exponential,
    fit_per_regime,
)
from .empirical import (
    BootstrapInterval,
    block_indices,
    bootstrap_difference,
)
from .features import PairTable, build_pair_table
from .field import REGIME_COUNT, Record
from .geometry import upper_triangle_pairs
from .learned import CorrelationPrediction, LearnedCorrelationModel
from .splits import HELD_OUT_SITES, Split, make_split

#: Names of the equation (39) models, in report order.
MODEL_NAMES: tuple[str, ...] = (
    "independent",
    "exponential",
    "per_regime",
    "learned",
)

#: Declared number of bootstrap replicates for the equation (41) margin.
MARGIN_REPLICATES: int = 400

#: Declared block length for that bootstrap, epochs.
MARGIN_BLOCK: int = 64


@dataclass(frozen=True)
class CellScore:
    """Scores for one split cell, equation (40).

    Attributes
    ----------
    cell:
        Cell name.
    rows:
        Number of pair-regime rows in it.
    joint_mae:
        Mean absolute error in joint unavailability, per model name.
    rho_mae:
        Mean absolute error in implied correlation, per model name.
    rho_truth_mae:
        Mean absolute error against the generator's TRUE correlation, per model
        name, or an empty mapping when no ground truth was supplied. This is
        the figure a synthetic record buys and a real one cannot.
    interval_coverage:
        Measured coverage of the learned model's interval against the implied
        correlation target, or ``nan`` where not applicable.
    """

    cell: str
    rows: int
    joint_mae: dict[str, float]
    rho_mae: dict[str, float]
    rho_truth_mae: dict[str, float] = field(default_factory=dict)
    interval_coverage: float = float("nan")


@dataclass(frozen=True)
class MarginResult:
    """Equation (41) for one hold-out cell against one baseline.

    Attributes
    ----------
    cell:
        Cell name.
    baseline:
        Baseline model name.
    margin:
        ``MAE(baseline) - MAE(learned)``; positive favours the learned model.
    bootstrap:
        Paired bootstrap interval of the same margin.
    reference_half_width:
        Half-width of the bootstrap interval on the empirical reference's own
        mean joint rate over the cell. Reported because it is the figure the
        specification names, and because it is the scale at which the reference
        itself is uncertain.
    """

    cell: str
    baseline: str
    margin: float
    bootstrap: BootstrapInterval
    reference_half_width: float

    @property
    def inside_interval(self) -> bool:
        """Whether the bootstrap interval on the margin contains zero.

        ``True`` means the two models are not distinguishable at the precision
        the empirical reference affords. That is a result, not a failure.
        """
        return self.bootstrap.contains(0.0)

    @property
    def inside_reference_half_width(self) -> bool:
        """Whether the margin is smaller than the reference's own half-width."""
        return abs(self.margin) <= self.reference_half_width


@dataclass
class Study:
    """Everything one record's comparison produced.

    Attributes
    ----------
    record_name:
        Which declared record.
    climatology:
        The estimated marginal climatology.
    split:
        The equation (29) split.
    table:
        The pair-regime table.
    exponential, per_regime:
        The fitted baselines.
    learned:
        The fitted learned model.
    predictions:
        Predicted correlation per model name, over ALL rows of the table.
    joint_predicted:
        Predicted joint unavailability per model name, over all rows.
    scores:
        One ``CellScore`` per cell name.
    margins:
        Equation (41), one per (hold-out cell, baseline) pair.
    seconds:
        Wall-clock seconds for the whole study, and for the learned fit alone.
    """

    record_name: str
    climatology: Climatology
    split: Split
    table: PairTable
    exponential: ExponentialModel
    per_regime: PerRegimeModel
    learned: LearnedCorrelationModel
    learned_prediction: CorrelationPrediction
    predictions: dict[str, np.ndarray]
    joint_predicted: dict[str, np.ndarray]
    scores: dict[str, CellScore]
    margins: tuple[MarginResult, ...]
    seconds: float
    learned_seconds: float
    strongest_baseline: str
    record: Record

    def record_for_nodes(self) -> Record:
        """The record this study was run on, for rebuilding its epoch nodes.

        Named rather than accessed as a bare attribute so that it is obvious at
        the call site that the generator's ground truth travels with it. No
        estimator in this package reads ``Record.latent`` or
        ``Record.threshold_true``; the validation scripts do.
        """
        return self.record


def row_nodes(
    table: PairTable, climatology: Climatology, record: Record
) -> tuple[EpochNodes, ...]:
    """Binned epoch nodes for every row of the table, built once.

    Every model in equation (39) is evaluated through the same nodes, so they
    are built here and shared. Building them per model would cost four times as
    much and, worse, would leave open the possibility that two models were
    scored through different nodes.

    The binning error is MEASURED in ``validation/validate_copula.py`` against
    the exact epoch-by-epoch evaluation of ``copula.pair_joint_exact``; it is
    three orders of magnitude below the bootstrap standard error of the
    reference, and the headline figures of this product are computed exactly
    rather than from nodes.
    """
    threshold = climatology.threshold_series()
    windows: dict[tuple[int, int], np.ndarray] = {}
    nodes: list[EpochNodes] = []
    for row in range(table.rows):
        key = (int(table.half[row]), int(table.regime[row]))
        if key not in windows:
            mask = (record.regime == key[1]) & table_half_mask(record, key[0])
            windows[key] = threshold[mask]
        window = windows[key]
        nodes.append(
            epoch_nodes(
                window[:, int(table.site_i[row])], window[:, int(table.site_j[row])]
            )
        )
    return tuple(nodes)


def joint_from_rho(nodes: tuple[EpochNodes, ...], rho: np.ndarray) -> np.ndarray:
    """Joint unavailability per row from a predicted correlation per row.

    The correlation is clamped into the copula's declared bracket before use. A
    tree ensemble is not constrained to return a correlation and the clamp is
    counted by ``CorrelationPrediction.clamp_count``.
    """
    values = np.asarray(rho, dtype=float)
    if values.shape != (len(nodes),):
        raise ValueError(
            f"rho must have shape ({len(nodes)},), got {values.shape}"
        )
    return np.array(
        [
            node.joint(float(np.clip(value, RHO_MIN, RHO_MAX)))
            for node, value in zip(nodes, values, strict=True)
        ],
        dtype=float,
    )


def table_half_mask(record: Record, half: int) -> np.ndarray:
    """Boolean epoch mask of time half ``half`` for ``record``."""
    midpoint = record.epochs // 2
    epoch = np.arange(record.epochs)
    return epoch < midpoint if half == 0 else epoch >= midpoint


def _mae(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(observed))))


def run_study(
    record: Record,
    record_name: str,
    held_out_sites: tuple[int, ...] = HELD_OUT_SITES,
    margin_replicates: int = MARGIN_REPLICATES,
    margin_block: int = MARGIN_BLOCK,
    margin_seed: int = 770001,
    true_correlation_rows: np.ndarray | None = None,
) -> Study:
    """Run the ordered comparison of equations (39) to (41) on one record.

    Parameters
    ----------
    record:
        The record to study.
    record_name:
        Label carried into the result.
    held_out_sites:
        Station indices held out, equation (31).
    margin_replicates, margin_block, margin_seed:
        Bootstrap settings for equation (41).
    true_correlation_rows:
        ``(rows,)`` generator truth per table row, for the ground-truth column
        of ``CellScore``. Supplied by validation scripts and by nothing else.

    Returns
    -------
    Study
    """
    import time

    started = time.perf_counter()
    pairs = upper_triangle_pairs(record.site_count)
    climatology = estimate_climatology(record.cloud)
    split = make_split(record.epochs, record.site_count, held_out_sites)
    table = build_pair_table(
        cloud=record.cloud,
        regime=record.regime,
        climatology=climatology,
        split=split,
        regime_count=REGIME_COUNT,
        pairs=pairs,
        sites=record.sites,
    )
    train = table.mask(0)
    if not np.any(train):
        raise ValueError("the split left no training rows")

    # Baselines first. Nothing learned has been fitted at this point.
    separation = table.separation_km()
    exponential = fit_exponential(separation[train], table.target[train])
    per_regime = fit_per_regime(
        separation[train], table.target[train], table.regime[train], REGIME_COUNT
    )

    learned_started = time.perf_counter()
    learned = LearnedCorrelationModel().fit(
        table.features[train], table.target[train]
    )
    learned_seconds = time.perf_counter() - learned_started
    prediction = learned.predict(table.features)

    predictions: dict[str, np.ndarray] = {
        "independent": np.zeros(table.rows, dtype=float),
        "exponential": exponential.predict(separation),
        "per_regime": per_regime.predict(separation, table.regime),
        "learned": prediction.median,
    }
    nodes = row_nodes(table, climatology, record)
    joint_predicted = {
        name: joint_from_rho(nodes, rho) for name, rho in predictions.items()
    }

    truth = (
        np.asarray(true_correlation_rows, dtype=float)
        if true_correlation_rows is not None
        else None
    )
    scores: dict[str, CellScore] = {}
    for code, name in ((0, "train"), (1, "held-out SITES"), (2, "held-out TIME")):
        mask = table.mask(code)
        rows = int(np.count_nonzero(mask))
        if rows == 0:
            continue
        scores[name] = CellScore(
            cell=name,
            rows=rows,
            joint_mae={
                model: _mae(joint_predicted[model][mask], table.observed[mask])
                for model in MODEL_NAMES
            },
            rho_mae={
                model: _mae(predictions[model][mask], table.target[mask])
                for model in MODEL_NAMES
            },
            rho_truth_mae=(
                {
                    model: _mae(predictions[model][mask], truth[mask])
                    for model in MODEL_NAMES
                }
                if truth is not None
                else {}
            ),
            interval_coverage=float(
                np.mean(prediction.covers(table.target)[mask])
            ),
        )

    strongest = min(
        ("exponential", "per_regime"),
        key=lambda name: scores["train"].joint_mae[name],
    )

    margins = _margins(
        record=record,
        table=table,
        joint_predicted=joint_predicted,
        replicates=margin_replicates,
        block=margin_block,
        seed=margin_seed,
    )
    return Study(
        record_name=record_name,
        climatology=climatology,
        split=split,
        table=table,
        exponential=exponential,
        per_regime=per_regime,
        learned=learned,
        learned_prediction=prediction,
        predictions=predictions,
        joint_predicted=joint_predicted,
        scores=scores,
        margins=margins,
        seconds=float(time.perf_counter() - started),
        learned_seconds=float(learned_seconds),
        strongest_baseline=strongest,
        record=record,
    )


def _margins(
    record: Record,
    table: PairTable,
    joint_predicted: dict[str, np.ndarray],
    replicates: int,
    block: int,
    seed: int,
) -> tuple[MarginResult, ...]:
    """Equation (41): the paired bootstrap on each hold-out cell's margin.

    What is resampled, precisely. A table row is one station pair in one
    seasonal regime of one time half, so its reference statistic is a mean over
    that window's epochs and the bootstrap resamples blocks WITHIN the window.
    Rows sharing a window share one set of draws, and every model is scored
    against the same resampled reference, so a difference of two models' errors
    is a difference of paired replicates as equation (22) requires.

    The replicates are processed in chunks of ``CHUNK`` because the full
    resampling index array over the reference record is large: 400 replicates of
    an eight-year three-hourly window against 85 pairs is a boolean array of
    about 99 megabytes, and the unchunked form over the whole record rather than
    the window exhausted 7.8 GiB during this product's build. The chunking
    changes no number: the draws are generated from one seeded generator per
    window in a fixed order.
    """
    chunk = 100
    results: list[MarginResult] = []
    for code, name in ((1, "held-out SITES"), (2, "held-out TIME")):
        mask = table.mask(code)
        if not np.any(mask):
            continue
        rows = np.where(mask)[0]
        replicate_error = {model: np.zeros(replicates) for model in MODEL_NAMES}
        replicate_reference_sum = np.zeros(replicates)
        windows: dict[tuple[int, int], list[int]] = {}
        for row in rows:
            windows.setdefault(
                (int(table.half[row]), int(table.regime[row])), []
            ).append(int(row))
        error_sum = {model: np.zeros(replicates) for model in MODEL_NAMES}
        for index, (key, window_rows) in enumerate(sorted(windows.items())):
            half, which = key
            epochs = np.where(
                (record.regime == which) & table_half_mask(record, half)
            )[0]
            both = np.column_stack(
                [
                    record.cloud[epochs, int(table.site_i[row])]
                    & record.cloud[epochs, int(table.site_j[row])]
                    for row in window_rows
                ]
            )
            draws = block_indices(
                epochs.size, block, replicates, seed + 1000 * code + index
            )
            for start in range(0, replicates, chunk):
                stop = min(start + chunk, replicates)
                reference = both[draws[start:stop]].mean(axis=1)
                replicate_reference_sum[start:stop] += reference.sum(axis=1)
                for model in MODEL_NAMES:
                    error_sum[model][start:stop] += np.abs(
                        reference - joint_predicted[model][window_rows][None, :]
                    ).sum(axis=1)
        total_rows = float(rows.size)
        for model in MODEL_NAMES:
            replicate_error[model] = error_sum[model] / total_rows
        reference_mean = replicate_reference_sum / total_rows
        tail = 0.025
        low, high = np.quantile(reference_mean, [tail, 1.0 - tail])
        for baseline in ("exponential", "per_regime"):
            observed_margin = float(
                np.mean(np.abs(joint_predicted[baseline][rows] - table.observed[rows]))
                - np.mean(
                    np.abs(joint_predicted["learned"][rows] - table.observed[rows])
                )
            )
            results.append(
                MarginResult(
                    cell=name,
                    baseline=baseline,
                    margin=observed_margin,
                    bootstrap=bootstrap_difference(
                        replicate_error[baseline],
                        replicate_error["learned"],
                        observed_margin,
                        block=block,
                    ),
                    reference_half_width=float(0.5 * (high - low)),
                )
            )
    return tuple(results)
