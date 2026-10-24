"""Score the predictors against each other, per regime, and report either way.

Metrics
-------
``median_abs_rel_error``
    Median of ``|predicted median - observed| / observed``, dimensionless. The
    median rather than the mean, for the same reason the delivery-time
    distribution is reported as quantiles: one heavy-tailed run would otherwise
    set the score.
``p90_abs_rel_error``
    The 90th percentile of the same quantity, so that a predictor which is
    usually right and occasionally very wrong cannot hide behind its median.
``coverage``
    Fraction of held-out rows whose observed delivery time fell inside the
    predicted interval. Compared against ``nominal``; the difference is the
    calibration error, and an interval that is not calibrated is reported as
    not calibrated rather than quietly widened.
``median_interval_width_ratio``
    Median of ``(upper - lower) / observed``, dimensionless. Coverage without a
    width is not a result: an interval from zero to infinity has perfect
    coverage.

The comparison is a hold-out comparison on rows generated from disjoint seed
ranges, and the analytic predictor never sees the training rows at all. That
asymmetry favours the learned model, which gets to see the generator; it is
stated here so that a win for the learned model is read with it in mind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import TARGET_COLUMN, targets
from .predictors import DeliveryPredictor


@dataclass(frozen=True)
class BenchmarkRow:
    """One predictor scored on one slice of held-out rows."""

    predictor: str
    regime: str
    rows: int
    median_abs_rel_error: float
    p90_abs_rel_error: float
    coverage: float
    nominal_coverage: float
    median_interval_width_ratio: float
    uses_training_data: bool

    @property
    def calibration_error(self) -> float:
        """``coverage - nominal_coverage``, dimensionless and signed."""
        return self.coverage - self.nominal_coverage

    @property
    def is_calibrated(self) -> bool:
        """True when coverage is within 0.03 of nominal.

        The threshold is declared here and not chosen after the measurement:
        0.03 is roughly three standard errors of a coverage estimate on a
        thousand held-out rows at a nominal 0.9, so a predictor inside it
        cannot be distinguished from calibrated at this sample size.
        """
        return abs(self.calibration_error) <= 0.03


def coverage_standard_error(nominal: float, rows: int) -> float:
    """Standard error of a coverage estimate, dimensionless.

    ``sqrt(q (1 - q) / m)`` for a binomial proportion. Reported beside every
    coverage number so that a deviation can be read against the noise of the
    estimate rather than against nothing.
    """
    if rows < 1:
        raise ValueError(f"rows must be at least 1; got {rows}")
    return float(np.sqrt(nominal * (1.0 - nominal) / rows))


def score(
    predictor: DeliveryPredictor,
    table: pd.DataFrame,
    regime: str,
    nominal: float = 0.90,
) -> BenchmarkRow:
    """Score one fitted predictor on ``table``."""
    if len(table) == 0:
        raise ValueError("cannot score a predictor on an empty table")
    observed = targets(table)
    if not np.all(np.isfinite(observed)):
        raise ValueError(
            f"column {TARGET_COLUMN!r} contains a non-finite value; score only completed runs"
        )
    lower, median, upper = predictor.predict_interval(table, nominal=nominal)
    relative = np.abs(median - observed) / observed
    inside = (observed >= lower) & (observed <= upper)
    width = (upper - lower) / observed
    return BenchmarkRow(
        predictor=predictor.name,
        regime=regime,
        rows=int(len(table)),
        median_abs_rel_error=float(np.median(relative)),
        p90_abs_rel_error=float(np.quantile(relative, 0.90)),
        coverage=float(np.mean(inside)),
        nominal_coverage=float(nominal),
        median_interval_width_ratio=float(np.median(width)),
        uses_training_data=bool(predictor.uses_training_data),
    )


def run_benchmark(
    predictors: list[DeliveryPredictor],
    train: pd.DataFrame,
    test: pd.DataFrame,
    nominal: float = 0.90,
    regimes: tuple[str, ...] | None = None,
) -> list[BenchmarkRow]:
    """Fit every predictor on ``train`` and score it on ``test``, per regime.

    The ``all`` slice is reported first, then each regime separately, because
    the per-regime split is where the specification's prediction lives: the
    closed form is exact in one regime and an extension in the other.
    """
    if regimes is None:
        regimes = tuple(sorted(test["regime"].unique()))
    rows: list[BenchmarkRow] = []
    for predictor in predictors:
        predictor.fit(train)
        rows.append(score(predictor, test, "all", nominal))
        for regime in regimes:
            slice_ = test[test["regime"] == regime]
            if len(slice_) == 0:  # pragma: no cover - defensive
                continue
            rows.append(score(predictor, slice_.reset_index(drop=True), regime, nominal))
    return rows


def calibration_table(
    predictor: DeliveryPredictor,
    table: pd.DataFrame,
    nominals: tuple[float, ...] = (0.50, 0.80, 0.90),
) -> list[tuple[float, float, float]]:
    """``(nominal, measured coverage, median width ratio)`` at each nominal level."""
    out = []
    observed = targets(table)
    for nominal in nominals:
        lower, _, upper = predictor.predict_interval(table, nominal=nominal)
        inside = (observed >= lower) & (observed <= upper)
        width = (upper - lower) / observed
        out.append((nominal, float(np.mean(inside)), float(np.median(width))))
    return out


def best_on(rows: list[BenchmarkRow], regime: str, metric: str = "median_abs_rel_error") -> str:
    """Name of the predictor with the lowest ``metric`` in ``regime``."""
    candidates = [row for row in rows if row.regime == regime]
    if not candidates:
        raise ValueError(f"no benchmark row for regime {regime!r}")
    return min(candidates, key=lambda row: getattr(row, metric)).predictor


def render_table(rows: list[BenchmarkRow]) -> list[str]:
    """Fixed-width lines for a report or a committed validation output."""
    header = (
        f"{'predictor':<18}{'regime':<20}{'rows':>6}{'med|rel err|':>14}"
        f"{'p90|rel err|':>14}{'coverage':>10}{'nominal':>9}{'width/obs':>11}{'trained':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.predictor:<18}{row.regime:<20}{row.rows:>6}"
            f"{row.median_abs_rel_error:>14.6f}{row.p90_abs_rel_error:>14.6f}"
            f"{row.coverage:>10.4f}{row.nominal_coverage:>9.2f}"
            f"{row.median_interval_width_ratio:>11.4f}"
            f"{str(row.uses_training_data):>9}"
        )
    return lines
