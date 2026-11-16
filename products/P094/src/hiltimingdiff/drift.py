"""Board drift over the campaign, fitted rather than bracketed, and its share.

The board got slower by 7.52 per cent while the campaign ran. The fixed
160x160 float64 matmul reference — data-independent, sharing no code with the
pipeline — measured a median of 141844.5 ns in the block before the first cell
and 152517.0 ns in the block after the last, a ratio of 1.0752408447278534
recorded in ``meta.json`` as ``calibration_drift_ratio``.

This is not a footnote. P093 measured its floor once, at the start, under the
highest foreign load of its session, and a published bound was falsified by
exactly this effect. Here there are **eleven** calibration blocks — ``floor_start``,
one before each of the nine cells, and ``floor_end`` — so the drift can be
*fitted over elapsed time* and the share of each stage's divergence ratio that
could be drift can be reported rather than argued about. SPEC §8 says ten
blocks; the campaign program emits eleven, and the specification's count is one
low.

Two things this module reports, both labelled, with the raw measured figure as
the headline
------------------------------------------------------------------------------

* the **raw measured** ``R[s]``, which is what the board did;
* a **drift-adjusted** variant, in which every cell's hardware times are
  divided by the fitted drift factor at that cell's mid-time, normalised so the
  factor is 1 at the start of the campaign.

The adjustment is a *per-cell* multiplicative factor and is therefore
**stage-independent by construction**. It can move the level of an individual
``R[s]``, and it cannot explain a difference *between* stages. That is a
structural statement about the adjustment, not a measurement, and the measured
spread before and after adjustment is reported so a reader can see it.

Confounding, checked rather than asserted
-----------------------------------------
The cells ran in randomised order under seed 20261115, permutation
``[1, 6, 0, 2, 8, 5, 4, 3, 7]``, so elapsed time is not confounded with frame
dimension. :func:`confounding` checks it: it regresses each cell's median total
frame time on ``order_index`` and on ``W`` separately, and reports the
correlation between ``order_index`` and ``W`` itself, which is the quantity the
randomisation was for.

Spearman's rho is computed on **tie-averaged midranks**, not on ordinal ranks.
The distinction is not pedantic and it cost this product a wrong published
number. ``numpy.argsort(numpy.argsort(x))`` is an ORDINAL rank: it breaks ties
by position in the array. ``W`` takes three values over nine cells -- two 48s,
five 64s and two 96s -- so an ordinal rank gives the five 64s five distinct
ranks instead of five copies of their midrank, and the resulting statistic is a
Pearson correlation on an arbitrary ordering rather than Spearman's rho. It is
not even well defined: on this campaign the ordinal form returns 0.85 when the
cells are in execution order and 0.5333333333333333 when they are in the
alphabetical order a ``groupby`` happens to produce, from identical data. The
midrank form returns 0.5809475019311126 either way, because a rank statistic
must be invariant under a permutation of the observations.

Units: nanoseconds for every time, seconds for ``elapsed_s``, dimensionless for
every ratio and coefficient of determination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bootstrap import BOOTSTRAP_SEED, CONFIDENCE, RESAMPLES
from .divergence import Divergence, PassUnits, divergence
from .traces import CALIBRATION_BLOCKS, Campaign


@dataclass(frozen=True)
class LinearFit:
    """Ordinary least squares of ``y`` on ``x``, with its explained variance."""

    slope: float
    intercept: float
    r_squared: float
    n: int
    x_label: str
    y_label: str

    def predict(self, x: np.ndarray | float) -> np.ndarray:
        """Fitted ``y`` at ``x``."""
        return self.intercept + self.slope * np.asarray(x, dtype=np.float64)


def linear_fit(
    x: np.ndarray, y: np.ndarray, *, x_label: str = "x", y_label: str = "y"
) -> LinearFit:
    """Least-squares straight line, with ``r_squared`` as explained variance.

    ``r_squared`` is ``1 - SSE/SST``, which for a straight line with an
    intercept equals the squared Pearson correlation. It is computed from the
    residuals rather than from the correlation so that the number reported is
    the fraction of variance this particular fit explains.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must be the same length: {xs.shape} vs {ys.shape}")
    if xs.size < 3:
        raise ValueError(f"a fit with a reported r_squared needs at least 3 points, got {xs.size}")
    if np.ptp(xs) == 0.0:
        raise ValueError(f"{x_label} is constant; no slope is defined")
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = ys - (intercept + slope * xs)
    total = ys - ys.mean()
    sst = float(np.sum(total**2))
    r_squared = float("nan") if sst == 0.0 else 1.0 - float(np.sum(residual**2)) / sst
    return LinearFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        n=int(xs.size),
        x_label=x_label,
        y_label=y_label,
    )


def midranks(values: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks of ``values``, 1-based, as Spearman's rho requires.

    Tied observations all receive the mean of the ranks they span, so the
    result is invariant under a permutation of the input -- which an ordinal
    rank is not, and which is the whole reason this function exists rather than
    ``numpy.argsort(numpy.argsort(x))``.

    Hand calculation, used as a test: ``[10, 20, 20, 30]`` has ranks
    ``[1, 2.5, 2.5, 4]``; the two 20s span ranks 2 and 3 and share their mean.

    Parameters
    ----------
    values:
        One-dimensional array. Non-finite entries are ranked by numpy's sort
        order and are not special-cased, because no caller in this package
        passes any.

    Returns
    -------
    numpy.ndarray
        Float ranks in ``[1, n]``, summing to ``n(n+1)/2`` exactly in exact
        arithmetic and to within rounding in floating point.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("midranks of an empty array are undefined")
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def spearman_rho(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman's rho: the Pearson correlation of the two series' midranks.

    Written out here rather than taken from a library so the tie treatment this
    product publishes is visible. ``tests/test_drift.py`` cross-checks it
    against ``scipy.stats.spearmanr``, which is a different implementation of
    the same definition and lives in the test extra rather than the runtime.

    Returns ``nan`` when either series is constant, because the correlation is
    then undefined rather than zero.
    """
    left = midranks(first)
    right = midranks(second)
    if left.shape != right.shape:
        raise ValueError(f"series must be the same length: {left.shape} vs {right.shape}")
    if left.size < 2:
        raise ValueError("Spearman's rho needs at least 2 observations")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


@dataclass(frozen=True)
class DriftFit:
    """The fitted board drift and the raw bracket it sits inside."""

    blocks: pd.DataFrame
    fit: LinearFit
    start_median_ns: float
    end_median_ns: float
    bracket_ratio: float
    meta_ratio: float
    fitted_ratio: float
    campaign_elapsed_s: float
    monotonic_blocks: int
    max_block_step_ratio: float

    def factor(self, elapsed_s: np.ndarray | float) -> np.ndarray:
        """Fitted drift factor at ``elapsed_s``, normalised to 1.0 at t = 0.

        A factor of 1.075 means the board is taking 7.5 per cent longer than it
        did at the start of the campaign for the same fixed work.
        """
        zero = self.fit.predict(0.0)
        if zero <= 0.0:
            raise ValueError("fitted drift reference at t=0 is non-positive")
        return self.fit.predict(elapsed_s) / zero


def calibration_blocks(campaign: Campaign) -> pd.DataFrame:
    """One row per calibration block: label, median ns, elapsed seconds.

    Raises
    ------
    ValueError
        If the committed calibration table does not carry exactly the eleven
        pinned block labels. The labels are literals so a campaign with a
        different calibration design is a refusal, not a different fit.
    """
    table = campaign.calibration
    present = tuple(table["which"].drop_duplicates())
    if set(present) != set(CALIBRATION_BLOCKS):
        raise ValueError(
            "calibration blocks are not the pinned set. Expected "
            f"{', '.join(CALIBRATION_BLOCKS)}; found {', '.join(present)}."
        )
    start = float(campaign.meta["t_campaign_start_mono"])
    rows = []
    for label in CALIBRATION_BLOCKS:
        block = table.loc[table["which"] == label]
        rows.append(
            {
                "which": label,
                "reps": int(len(block)),
                "median_ns": float(np.median(block["ns"].to_numpy(dtype=np.float64))),
                "t_mono": float(block["t_mono"].iloc[0]),
                "elapsed_s": float(block["t_mono"].iloc[0]) - start,
            }
        )
    return pd.DataFrame(rows)


def fit_drift(campaign: Campaign) -> DriftFit:
    """Fit the board's drift reference against elapsed campaign time."""
    blocks = calibration_blocks(campaign)
    fit = linear_fit(
        blocks["elapsed_s"].to_numpy(),
        blocks["median_ns"].to_numpy(),
        x_label="elapsed_s",
        y_label="calibration_median_ns",
    )
    start = float(blocks.loc[blocks["which"] == "floor_start", "median_ns"].iloc[0])
    end = float(blocks.loc[blocks["which"] == "floor_end", "median_ns"].iloc[0])
    elapsed = float(campaign.meta["elapsed_s"])
    medians = blocks["median_ns"].to_numpy(dtype=np.float64)
    steps = medians[1:] / medians[:-1]
    return DriftFit(
        blocks=blocks,
        fit=fit,
        start_median_ns=start,
        end_median_ns=end,
        bracket_ratio=end / start,
        meta_ratio=float(campaign.meta["calibration_drift_ratio"]),
        fitted_ratio=float(fit.predict(elapsed) / fit.predict(0.0)),
        campaign_elapsed_s=elapsed,
        monotonic_blocks=int(np.count_nonzero(np.diff(medians) > 0.0)),
        max_block_step_ratio=float(steps.max()),
    )


def cell_drift_factors(campaign: Campaign, drift: DriftFit) -> dict[str, float]:
    """Fitted drift factor at each cell's mid-time, keyed by cell name."""
    start = float(campaign.meta["t_campaign_start_mono"])
    factors: dict[str, float] = {}
    for row in campaign.runs.itertuples():
        middle = (float(row.t0_mono) + float(row.t1_mono)) / 2.0 - start
        factors[str(row.cell)] = float(drift.factor(middle))
    return factors


def drift_adjusted_units(units: PassUnits, factors: dict[str, float]) -> PassUnits:
    """Pass units with each cell's hardware times divided by its drift factor.

    The declared operation counts are untouched: an operation count does not
    drift. Missing a factor for a cell is a refusal rather than a factor of 1,
    because silently leaving a cell unadjusted would understate the adjustment.
    """
    missing = sorted({cell for cell in units.cells if cell not in factors})
    if missing:
        raise ValueError(
            f"no fitted drift factor for cell(s) {', '.join(missing)}; "
            "a cell is never left unadjusted silently"
        )
    scale = np.asarray([factors[cell] for cell in units.cells], dtype=np.float64)
    if np.any(scale <= 0.0):
        raise ValueError("a fitted drift factor is non-positive")
    return PassUnits(
        stages=units.stages,
        cells=units.cells,
        pass_index=units.pass_index,
        hw=units.hw / scale[:, None],
        ops=units.ops,
    )


@dataclass(frozen=True)
class DriftComparison:
    """Raw measured divergence beside its drift-adjusted variant."""

    raw: Divergence
    adjusted: Divergence
    factors: dict[str, float]
    drift: DriftFit

    @property
    def spread_shift(self) -> float:
        """``adjusted spread - raw spread``, dimensionless."""
        return self.adjusted.spread - self.raw.spread

    @property
    def spread_relative_shift(self) -> float:
        """Spread shift as a fraction of the raw spread."""
        return self.spread_shift / self.raw.spread

    def ratio_shift(self) -> dict[str, float]:
        """Per-stage ``(adjusted R - raw R) / raw R``, dimensionless."""
        return {
            stage: (self.adjusted.ratio[stage] - self.raw.ratio[stage])
            / self.raw.ratio[stage]
            for stage in self.raw.stages
        }

    @property
    def max_abs_ratio_shift(self) -> float:
        """Largest absolute per-stage relative shift in ``R``."""
        return float(max(abs(value) for value in self.ratio_shift().values()))

    def frame(self) -> pd.DataFrame:
        """Raw and adjusted ``R`` side by side, per stage, in pipeline order."""
        shifts = self.ratio_shift()
        return pd.DataFrame(
            [
                {
                    "stage": stage,
                    "R_raw_measured": self.raw.ratio[stage],
                    "R_drift_adjusted": self.adjusted.ratio[stage],
                    "relative_shift": shifts[stage],
                }
                for stage in self.raw.stages
            ]
        )


def drift_comparison(
    campaign: Campaign,
    units: PassUnits,
    ns_per_op: float,
    *,
    confidence: float = CONFIDENCE,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> DriftComparison:
    """Raw measured divergence and its drift-adjusted variant, both labelled.

    The raw measured figure is the headline. The adjusted one answers "how much
    of this could be drift" and no more than that.
    """
    drift = fit_drift(campaign)
    factors = cell_drift_factors(campaign, drift)
    raw = divergence(
        units, ns_per_op, confidence=confidence, resamples=resamples, seed=seed
    )
    adjusted = divergence(
        drift_adjusted_units(units, factors),
        ns_per_op,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return DriftComparison(raw=raw, adjusted=adjusted, factors=factors, drift=drift)


@dataclass(frozen=True)
class Confounding:
    """Whether elapsed time is confounded with frame dimension. It is not."""

    cells: tuple[str, ...]
    order_index: np.ndarray
    width: np.ndarray
    median_total_ns: np.ndarray
    on_order: LinearFit
    on_width: LinearFit
    order_width_correlation: float
    #: Spearman's rho on TIE-AVERAGED MIDRANKS. See :func:`spearman_rho`.
    order_width_spearman: float

    @property
    def width_dominates(self) -> bool:
        """Whether width explains more of the between-cell variance than order."""
        return self.on_width.r_squared > self.on_order.r_squared


def confounding(campaign: Campaign, units: PassUnits) -> Confounding:
    """Regress each cell's median total frame time on order and on width.

    Two separate single-variable regressions, as SPEC §8's randomisation claim
    requires: if the randomisation worked, ``order_index`` and ``W`` are close
    to uncorrelated and width carries the between-cell variance that elapsed
    time does not.
    """
    table = pd.DataFrame(units.hw, columns=list(units.stages))
    table["cell"] = list(units.cells)
    totals = table.groupby("cell").sum(numeric_only=True).sum(axis=1)
    per_pass_total = table.groupby("cell")[list(units.stages)].median().sum(axis=1)
    del totals
    cells = tuple(per_pass_total.index)
    order = np.asarray([campaign.order_index(cell) for cell in cells], dtype=np.float64)
    width = np.asarray([campaign.cell_width(cell) for cell in cells], dtype=np.float64)
    median_total = per_pass_total.to_numpy(dtype=np.float64)
    correlation = float(np.corrcoef(order, width)[0, 1])
    # Tie-averaged midranks, NOT argsort(argsort(x)). W has five tied 64s and an
    # ordinal rank would make the published statistic depend on the row order
    # this groupby happens to produce. See the module docstring.
    spearman = spearman_rho(order, width)
    return Confounding(
        cells=cells,
        order_index=order,
        width=width,
        median_total_ns=median_total,
        on_order=linear_fit(
            order, median_total, x_label="order_index", y_label="cell_median_total_ns"
        ),
        on_width=linear_fit(
            width, median_total, x_label="w", y_label="cell_median_total_ns"
        ),
        order_width_correlation=correlation,
        order_width_spearman=spearman,
    )
