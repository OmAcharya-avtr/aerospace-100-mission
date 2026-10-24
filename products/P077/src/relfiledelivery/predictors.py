"""Three delivery-time predictors, scored against each other on the same rows.

The specification for this product predicts in writing that the analytic model
wins wherever its assumptions hold and that the learned model earns its place
only in the correlated-outage regime, if at all. The three predictors below
exist so that prediction can be measured rather than argued about, and
:mod:`relfiledelivery.benchmark` reports whichever way it comes out.

``AnalyticPredictor``
    The closed form of :mod:`relfiledelivery.analytic`, divided by the link
    availability to extend it past its own assumptions. Its interval comes
    from the exact negative-binomial transmission-count quantile and the exact
    round-count quantile. **It does not look at the training data at all**, so
    it has no capacity to overfit and nothing to tune. Every number it produces
    is a consequence of the model in ``analytic.py``.

``MeanLookupPredictor``
    A table of measured quantiles binned by mode, loss probability and
    availability. This is what an engineer with a spreadsheet of past transfers
    actually does, and it is the baseline that matters: a learned model that
    cannot beat a lookup table has learned nothing a lookup table does not
    already encode. Despite the name it reports the measured *median*, not the
    mean, because a mean over an intermittent link is not a prediction.

``LearnedQuantilePredictor``
    Gradient-boosted quantile regression (Friedman 2001, "Greedy function
    approximation: a gradient boosting machine", *Annals of Statistics* 29(5),
    1189--1232) on the base-10 logarithm of the delivery time, one model per
    quantile level. Quantiles commute with a monotone transform, so the
    exponentiated prediction is the quantile of the delivery time and not an
    approximation to it. The interval between the 5th and 95th percentile
    models is the model's uncertainty output; whether it is *calibrated* is a
    measured quantity and is reported against nominal either way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .analytic import extended_delivery_time_quantile, extended_expected_delivery_time
from .dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    feature_matrix,
    scenario_from_row,
    targets,
)
from .modes import TransferMode

DEFAULT_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class DeliveryPredictor(ABC):
    """Predicts delivery-time quantiles for a table of scenarios."""

    name: str = "predictor"
    uses_training_data: bool = True

    @abstractmethod
    def fit(self, table: pd.DataFrame) -> DeliveryPredictor:
        """Fit on a table of completed runs. Returns ``self``."""

    @abstractmethod
    def predict_quantiles(
        self, table: pd.DataFrame, levels: tuple[float, ...] = DEFAULT_LEVELS
    ) -> dict[float, np.ndarray]:
        """Predicted delivery time in seconds at each level, one array per level."""

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        """Median prediction in seconds."""
        return self.predict_quantiles(table, (0.50,))[0.50]

    def predict_interval(
        self, table: pd.DataFrame, nominal: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lower, median, upper)`` for a central interval of mass ``nominal``.

        Seconds. The interval is the model's uncertainty output; its measured
        coverage is reported by :mod:`relfiledelivery.benchmark`.
        """
        if not 0.0 < nominal < 1.0:
            raise ValueError(f"nominal must lie in (0, 1); got {nominal}")
        lo_level = round((1.0 - nominal) / 2.0, 6)
        hi_level = round(1.0 - lo_level, 6)
        got = self.predict_quantiles(table, (lo_level, 0.50, hi_level))
        return got[lo_level], got[0.50], got[hi_level]


def _column(table: pd.DataFrame, name: str) -> np.ndarray:
    if name not in table.columns:
        raise ValueError(f"table is missing column {name!r}")
    return np.asarray(table[name], dtype=float)


class AnalyticPredictor(DeliveryPredictor):
    """Closed form, extended past its own assumptions and scored as it stands.

    The point prediction is
    :func:`relfiledelivery.analytic.extended_delivery_time_quantile` at 0.50,
    and the interval ends are the same function at the interval's levels: the
    closed form on the outage-free link, divided by the outage duty cycle, laid
    out over the declared contact windows. The quantile inside it is built from
    two exact marginals -- the negative-binomial transmission count and the
    closed-form round count -- evaluated at the same probability; both
    marginals are exact, evaluating two dependent marginals at the same
    probability and adding them is not, and the consequence is visible in the
    measured coverage.

    Fitting is a no-op. This predictor is a formula and has no parameters.

    Parameters
    ----------
    point:
        ``"median"`` (the default) evaluates the quantile expression at 0.50,
        so the point prediction is the analytic *median*. ``"mean"`` uses the
        expectation ``n / (1 - p)`` and ``E[R]`` instead. Both are benchmarked,
        because the expectation is the number a closed form naturally produces
        and the median is the number a point prediction should be, and the gap
        between them over a multimodal delivery-time distribution is a result
        of this package rather than a detail.
    """

    uses_training_data = False

    def __init__(self, point: str = "median") -> None:
        if point not in ("median", "mean"):
            raise ValueError(f"point must be 'median' or 'mean'; got {point!r}")
        self.point = point
        self.name = "analytic" if point == "median" else "analytic_mean"

    def fit(self, table: pd.DataFrame) -> AnalyticPredictor:
        """No-op. Recorded explicitly so the benchmark can state it."""
        return self

    def predict_quantiles(
        self, table: pd.DataFrame, levels: tuple[float, ...] = DEFAULT_LEVELS
    ) -> dict[float, np.ndarray]:
        # The scenario is rebuilt from its seed and regime rather than
        # reassembled from the stored feature columns, so the predictor calls
        # exactly the functions in analytic.py and cannot drift from them. The
        # rebuild is deterministic and costs no simulation.
        if "seed" not in table.columns or "regime" not in table.columns:
            raise ValueError(
                "table is missing column 'seed' or 'regime'; AnalyticPredictor rebuilds the "
                "scenario from them so that it calls the same closed form the rest of the "
                "package does"
            )
        scenarios = [scenario_from_row(table.iloc[index]) for index in range(len(table))]
        out: dict[float, np.ndarray] = {}
        for level in levels:
            values = np.empty(len(table), dtype=float)
            for index, scenario in enumerate(scenarios):
                reference_mode = (
                    TransferMode.DEFERRED_NAK
                    if scenario.mode is TransferMode.IMMEDIATE_NAK
                    else scenario.mode
                )
                if level == 0.50 and self.point == "mean":
                    values[index] = extended_expected_delivery_time(
                        scenario.spec, scenario.link, reference_mode
                    )
                else:
                    values[index] = extended_delivery_time_quantile(
                        scenario.spec, scenario.link, level
                    )
            out[level] = values
        return out


@dataclass
class _Bin:
    quantiles: dict[float, float] = field(default_factory=dict)
    count: int = 0


class MeanLookupPredictor(DeliveryPredictor):
    """Measured quantiles binned by mode, loss probability and availability.

    The bin edges are declared, not learned. A bin with fewer than
    ``min_count`` training rows falls back to the global table, and how often
    that happens is reported, because a lookup table that silently falls back
    everywhere is not the baseline it appears to be.
    """

    name = "mean_lookup"

    def __init__(
        self,
        loss_edges: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 1.0),
        availability_edges: tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 1.01),
        min_count: int = 12,
    ) -> None:
        self.loss_edges = loss_edges
        self.availability_edges = availability_edges
        self.min_count = int(min_count)
        self._bins: dict[tuple[int, int, int], _Bin] = {}
        self._global = _Bin()
        self.fallback_count = 0
        self.lookup_count = 0

    def _key(self, loss: float, availability: float, immediate: float) -> tuple[int, int, int]:
        loss_bin = int(np.searchsorted(self.loss_edges, loss, side="right")) - 1
        avail_bin = int(np.searchsorted(self.availability_edges, availability, side="right")) - 1
        return (
            max(0, min(loss_bin, len(self.loss_edges) - 2)),
            max(0, min(avail_bin, len(self.availability_edges) - 2)),
            int(round(immediate)),
        )

    def fit(self, table: pd.DataFrame) -> MeanLookupPredictor:
        """Record per-bin empirical quantiles of the delivery time, in seconds.

        The prediction is scale-free in the file size and the link rate, which
        the bin key does not carry, so this baseline is deliberately weak in a
        way a spreadsheet of past transfers is also weak. It is reported as the
        baseline an engineer actually has, not as a strong competitor.
        """
        loss = _column(table, "pdu_loss_probability")
        availability = _column(table, "availability")
        immediate = _column(table, "mode_is_immediate_nak")
        clean = 10.0 ** _column(table, "log10_clean_pass_s")
        observed = targets(table)
        # Normalised by the clean-pass time so that the table carries a
        # dimensionless inflation factor rather than an absolute second count,
        # which would make it useless on a file of a different size.
        ratio = observed / clean

        buckets: dict[tuple[int, int, int], list[float]] = {}
        for index in range(len(table)):
            key = self._key(loss[index], availability[index], immediate[index])
            buckets.setdefault(key, []).append(float(ratio[index]))
        self._bins = {}
        for key, values in buckets.items():
            array = np.asarray(values, dtype=float)
            self._bins[key] = _Bin(
                quantiles={
                    level: float(np.quantile(array, level)) for level in DEFAULT_LEVELS
                },
                count=array.size,
            )
        self._global = _Bin(
            quantiles={
                level: float(np.quantile(ratio, level)) for level in DEFAULT_LEVELS
            },
            count=int(ratio.size),
        )
        self.fallback_count = 0
        self.lookup_count = 0
        return self

    def predict_quantiles(
        self, table: pd.DataFrame, levels: tuple[float, ...] = DEFAULT_LEVELS
    ) -> dict[float, np.ndarray]:
        if not self._global.quantiles:
            raise RuntimeError("MeanLookupPredictor.predict_quantiles called before fit")
        for level in levels:
            if level not in self._global.quantiles:
                raise ValueError(
                    f"level {level} was not tabulated; fitted levels are "
                    f"{sorted(self._global.quantiles)}"
                )
        loss = _column(table, "pdu_loss_probability")
        availability = _column(table, "availability")
        immediate = _column(table, "mode_is_immediate_nak")
        clean = 10.0 ** _column(table, "log10_clean_pass_s")

        out = {level: np.empty(len(table), dtype=float) for level in levels}
        self.fallback_count = 0
        self.lookup_count = 0
        for index in range(len(table)):
            key = self._key(loss[index], availability[index], immediate[index])
            bucket = self._bins.get(key)
            if bucket is None or bucket.count < self.min_count:
                bucket = self._global
                self.fallback_count += 1
            else:
                self.lookup_count += 1
            for level in levels:
                out[level][index] = bucket.quantiles[level] * clean[index]
        return out


class LearnedQuantilePredictor(DeliveryPredictor):
    """Gradient-boosted quantile regression on ``log10`` delivery time.

    One :class:`sklearn.ensemble.GradientBoostingRegressor` per quantile level
    with ``loss="quantile"``. Independent per-level models are not guaranteed to
    be monotone in the level, so the predictions are sorted across levels
    before they are returned and the number of crossings that had to be sorted
    is recorded in :attr:`crossing_count`. Hiding a crossing without counting
    it would make an uncalibrated interval look tidy.
    """

    name = "learned_quantile"

    def __init__(
        self,
        levels: tuple[float, ...] = DEFAULT_LEVELS,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.06,
        min_samples_leaf: int = 12,
        subsample: float = 0.9,
        random_state: int = 20261024,
    ) -> None:
        self.levels = tuple(levels)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = int(random_state)
        self.models: dict[float, GradientBoostingRegressor] = {}
        self.crossing_count = 0
        self.n_train = 0

    def fit(self, table: pd.DataFrame) -> LearnedQuantilePredictor:
        """Fit one quantile model per declared level on ``log10`` seconds."""
        x = feature_matrix(table)
        y = np.log10(targets(table))
        if not np.all(np.isfinite(y)):
            raise ValueError("training targets contain a non-finite delivery time")
        self.models = {}
        for level in self.levels:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=level,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            model.fit(x, y)
            self.models[level] = model
        self.n_train = int(len(table))
        return self

    def predict_quantiles(
        self, table: pd.DataFrame, levels: tuple[float, ...] = DEFAULT_LEVELS
    ) -> dict[float, np.ndarray]:
        if not self.models:
            raise RuntimeError("LearnedQuantilePredictor.predict_quantiles called before fit")
        missing = [level for level in levels if level not in self.models]
        if missing:
            raise ValueError(
                f"levels {missing} were not fitted; fitted levels are {sorted(self.models)}"
            )
        x = feature_matrix(table)
        ordered = sorted(levels)
        stack = np.vstack([10.0 ** self.models[level].predict(x) for level in ordered])
        crossings = int(np.sum(np.diff(stack, axis=0) < 0.0))
        self.crossing_count = crossings
        stack = np.sort(stack, axis=0)
        return {level: stack[position] for position, level in enumerate(ordered)}

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature column order the models were fitted on."""
        return FEATURE_COLUMNS

    def feature_importances(self, level: float = 0.50) -> dict[str, float]:
        """Impurity-based importances of the model at ``level``, dimensionless.

        Impurity importances are biased towards high-cardinality features
        (Strobl et al. 2007, *BMC Bioinformatics* 8:25) and are reported here as
        a description of the fitted model, not as evidence about the link.
        """
        if level not in self.models:
            raise ValueError(f"level {level} was not fitted")
        values = self.models[level].feature_importances_
        return dict(zip(FEATURE_COLUMNS, (float(v) for v in values), strict=True))


def build_predictors(random_state: int = 20261024) -> list[DeliveryPredictor]:
    """The three predictors the benchmark scores, in reporting order."""
    return [
        AnalyticPredictor(point="median"),
        AnalyticPredictor(point="mean"),
        MeanLookupPredictor(),
        LearnedQuantilePredictor(random_state=random_state),
    ]


__all__ = [
    "DEFAULT_LEVELS",
    "TARGET_COLUMN",
    "AnalyticPredictor",
    "DeliveryPredictor",
    "LearnedQuantilePredictor",
    "MeanLookupPredictor",
    "build_predictors",
]
