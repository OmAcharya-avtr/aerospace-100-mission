"""Split, Mondrian and weighted conformal prediction intervals.

All three use the absolute-residual conformity score

    V_i = |y_i - yhat(x_i)|                                        [Wh]

and return the symmetric interval ``yhat +/- q``. Asymmetric and
locally-normalised scores are not implemented; see the Limitations section of
README.md.

References
----------
- Split (inductive) conformal: Papadopoulos, H., Proedrou, K., Vovk, V. and
  Gammerman, A., "Inductive Confidence Machines for Regression", *ECML* 2002;
  Lei, J. et al., *JASA* 113(523), 2018.
- Mondrian (class-conditional) conformal: Vovk, V., Lindsay, D., Nouretdinov,
  I. and Gammerman, A., "Mondrian Confidence Machine", 2003; Vovk, V.,
  Gammerman, A. and Shafer, G., *Algorithmic Learning in a Random World*,
  Springer, 2005, chapter 4.
- Weighted conformal under covariate shift: Tibshirani, R.J., Barber, R.F.,
  Candes, E.J. and Ramdas, A., "Conformal Prediction Under Covariate Shift",
  *NeurIPS* 32, 2019.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .bounds import CoverageBound, split_conformal_coverage_bound

QUANTILE_RELATIVE_TOLERANCE = 1e-12
"""Relative slack applied to the weighted-quantile threshold.

``(n + 1) * (1 - alpha)`` evaluated in binary floating point can land one ulp
above the integer it should equal, and the selected order statistic is then one
rank too high, which silently makes every interval conservative. A demonstrated
case is ``n = 24, alpha = 0.44``: ``25 * (1 - 0.44) == 14.000000000000002``, so
an unguarded ``ceil`` returns 15 instead of 14. A search over
``n`` in [5, 2000) and ``alpha`` in {0.001, ..., 0.499} found no such case for
``alpha`` of 0.1, 0.05, 0.02 or 0.01, so the guard matters only for unusual
levels; it is kept because being wrong by one rank is invisible in the output.
Both the hazard and the guard are checked in ``tests/test_known_answers.py``.
"""


def conformal_rank(n_calibration: int, alpha: float) -> int:
    """``k = ceil((n + 1)(1 - alpha))`` with the floating-point guard applied.

    Parameters
    ----------
    n_calibration:
        Number of calibration scores, positive.
    alpha:
        Miscoverage level in (0, 1).

    Returns
    -------
    int
        The order statistic index, 1-based.
    """
    if n_calibration <= 0:
        raise ValueError(f"n_calibration must be > 0, got {n_calibration}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return int(math.ceil(round((n_calibration + 1) * (1.0 - alpha), 10)))


def absolute_residual_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Conformity score ``|y - yhat|`` [same units as ``y``]."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y.shape} and {p.shape}")
    if y.size == 0:
        raise ValueError("y_true must be non-empty")
    return np.abs(y - p)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    level: float,
    *,
    tail_weight: float = 0.0,
) -> float:
    """Smallest value whose weighted cumulative mass reaches ``level``.

    Implements ``inf{ v : sum_i p_i 1[V_i <= v] >= level }`` with
    ``p_i = w_i / (sum_j w_j + tail_weight)``. ``tail_weight`` is the point
    mass placed at ``+inf``, which is how weighted conformal accounts for the
    unobserved test score (Tibshirani et al. 2019, eq. 7).

    Parameters
    ----------
    values:
        Conformity scores, shape ``(n,)``, any units.
    weights:
        Non-negative weights, shape ``(n,)``.
    level:
        Target cumulative mass in (0, 1).
    tail_weight:
        Non-negative mass at ``+inf``.

    Returns
    -------
    float
        The quantile, or ``inf`` when the level is unattainable because the
        tail mass alone exceeds ``1 - level``.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.shape != w.shape:
        raise ValueError(
            f"values and weights must have the same shape, got {v.shape} and {w.shape}"
        )
    if v.size == 0:
        raise ValueError("values must be non-empty")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    if tail_weight < 0.0:
        raise ValueError(f"tail_weight must be >= 0, got {tail_weight}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    total = w.sum() + tail_weight
    if total <= 0.0:
        raise ValueError("total weight must be > 0")
    order = np.argsort(v, kind="stable")
    cumulative = np.cumsum(w[order])
    threshold = level * total * (1.0 - QUANTILE_RELATIVE_TOLERANCE)
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    if index >= v.size:
        return math.inf
    return float(v[order][index])


class Predictor(Protocol):
    """Anything with a ``predict(features) -> ndarray`` method."""

    def predict(self, features: np.ndarray) -> np.ndarray: ...  # pragma: no cover


@dataclass(frozen=True)
class Interval:
    """A prediction interval.

    Attributes
    ----------
    lower, point, upper:
        Arrays of shape ``(n,)`` [Wh].
    """

    lower: np.ndarray
    point: np.ndarray
    upper: np.ndarray

    @property
    def width(self) -> np.ndarray:
        """``upper - lower`` [Wh], ``inf`` where the interval is unbounded."""
        return self.upper - self.lower

    def covers(self, y_true: np.ndarray) -> np.ndarray:
        """Elementwise ``lower <= y <= upper`` [bool]."""
        y = np.asarray(y_true, dtype=float).ravel()
        if y.shape != self.point.shape:
            raise ValueError("y_true must have the same shape as the interval")
        return (y >= self.lower) & (y <= self.upper)


class SplitConformal:
    """Marginal split conformal prediction with absolute-residual scores.

    Parameters
    ----------
    alpha:
        Miscoverage level in (0, 1). ``0.1`` gives a nominal 90 % interval.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self._scores: np.ndarray | None = None
        self._quantile: float | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> SplitConformal:
        """Store the calibration scores and compute the conformal quantile."""
        scores = absolute_residual_score(y_true, y_pred)
        rank = conformal_rank(scores.size, self.alpha)
        if rank > scores.size:
            bound_msg = split_conformal_coverage_bound  # defer to its message
            try:
                bound_msg(scores.size, self.alpha)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        self._scores = np.sort(scores)
        self._quantile = float(self._scores[rank - 1])
        return self

    @property
    def is_calibrated(self) -> bool:
        """``True`` once :meth:`calibrate` has run."""
        return self._quantile is not None

    @property
    def quantile(self) -> float:
        """The conformal half-width ``q`` [Wh]."""
        if self._quantile is None:
            raise RuntimeError("call calibrate(...) before reading the quantile")
        return self._quantile

    @property
    def n_calibration(self) -> int:
        """Number of calibration scores ``n``."""
        if self._scores is None:
            raise RuntimeError("call calibrate(...) first")
        return int(self._scores.size)

    @property
    def coverage_bound(self) -> CoverageBound:
        """The finite-sample bound for this calibration size and ``alpha``."""
        return split_conformal_coverage_bound(self.n_calibration, self.alpha)

    def interval(self, y_pred: np.ndarray) -> Interval:
        """Symmetric interval ``yhat +/- q`` [Wh]."""
        p = np.asarray(y_pred, dtype=float).ravel()
        q = self.quantile
        return Interval(lower=p - q, point=p, upper=p + q)


def tercile_edges(values: np.ndarray) -> np.ndarray:
    """The two interior edges that split ``values`` into terciles [same units].

    Declared on the calibration predictions and then applied unchanged to the
    test predictions, so the Mondrian taxonomy does not depend on the test
    sample.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 3:
        raise ValueError(f"need at least 3 values to form terciles, got {v.size}")
    return np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in ``[0, len(edges)]`` for each value [int].

    Bins are half-open upward: with edges ``(e0, e1)`` the bins are
    ``(-inf, e0)``, ``[e0, e1)`` and ``[e1, inf)``, so a value sitting exactly
    on an edge falls in the upper bin.
    """
    v = np.asarray(values, dtype=float).ravel()
    e = np.asarray(edges, dtype=float).ravel()
    if e.size == 0:
        raise ValueError("edges must be non-empty")
    if np.any(np.diff(e) <= 0.0):
        raise ValueError("edges must be strictly increasing")
    return np.searchsorted(e, v, side="right").astype(int)


class MondrianConformal:
    """Class-conditional (Mondrian) split conformal, one quantile per bin.

    The taxonomy is supplied by the caller as an integer bin index per point.
    Each bin gets its own calibration sample, its own quantile and its own
    finite-sample bound, so a bin with few points is honestly wider rather
    than quietly borrowing strength from the others.

    Parameters
    ----------
    alpha:
        Miscoverage level in (0, 1).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self._quantiles: dict[int, float] = {}
        self._counts: dict[int, int] = {}

    def calibrate(
        self, y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray
    ) -> MondrianConformal:
        """Compute one conformal quantile per bin.

        Raises
        ------
        ValueError
            If any bin holds too few points for the requested ``alpha``. The
            message names the bin and the minimum size.
        """
        scores = absolute_residual_score(y_true, y_pred)
        b = np.asarray(bins).ravel()
        if b.shape != scores.shape:
            raise ValueError(f"bins must have shape {scores.shape}, got {b.shape}")
        self._quantiles = {}
        self._counts = {}
        for key in np.unique(b):
            block = np.sort(scores[b == key])
            rank = conformal_rank(block.size, self.alpha)
            if rank > block.size:
                needed = math.ceil(1.0 / self.alpha) - 1
                raise ValueError(
                    f"bin {int(key)} holds {block.size} calibration points, too few for "
                    f"alpha={self.alpha}: the required order statistic is {rank}. "
                    f"Each bin needs at least {needed} points, or use a coarser taxonomy."
                )
            self._quantiles[int(key)] = float(block[rank - 1])
            self._counts[int(key)] = int(block.size)
        return self

    @property
    def quantiles(self) -> dict[int, float]:
        """Per-bin conformal half-widths [Wh]."""
        if not self._quantiles:
            raise RuntimeError("call calibrate(...) first")
        return dict(self._quantiles)

    @property
    def counts(self) -> dict[int, int]:
        """Per-bin calibration counts."""
        if not self._counts:
            raise RuntimeError("call calibrate(...) first")
        return dict(self._counts)

    def coverage_bounds(self) -> dict[int, CoverageBound]:
        """Per-bin finite-sample coverage bounds."""
        return {k: split_conformal_coverage_bound(n, self.alpha) for k, n in self.counts.items()}

    def interval(self, y_pred: np.ndarray, bins: np.ndarray) -> Interval:
        """Symmetric per-bin interval [Wh].

        Raises
        ------
        KeyError
            If a test bin was never seen during calibration. Silently falling
            back to a pooled quantile would void the conditional guarantee,
            so it is refused.
        """
        p = np.asarray(y_pred, dtype=float).ravel()
        b = np.asarray(bins).ravel()
        if b.shape != p.shape:
            raise ValueError(f"bins must have shape {p.shape}, got {b.shape}")
        table = self.quantiles
        unseen = sorted({int(k) for k in np.unique(b)} - set(table))
        if unseen:
            raise KeyError(
                f"test bins {unseen} have no calibration quantile; a Mondrian predictor "
                "cannot produce a conditional interval for an unseen category"
            )
        q = np.array([table[int(k)] for k in b], dtype=float)
        return Interval(lower=p - q, point=p, upper=p + q)


class WeightedConformal:
    """Weighted split conformal for a declared covariate shift.

    Each calibration point carries a likelihood-ratio weight ``w(x_i)`` and
    each test point carries ``w(x)``, which enters its own normalisation and
    places mass ``w(x) / (sum_i w(x_i) + w(x))`` at ``+inf``. When that mass
    exceeds ``alpha`` the interval for that point is genuinely unbounded, and
    this class reports it as ``inf`` rather than truncating.

    The guarantee holds **only if the weights are the true likelihood ratio.**
    The breaking-point sweep in :mod:`conformalband.audit` measures what
    happens when they are not.

    Parameters
    ----------
    alpha:
        Miscoverage level in (0, 1).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self._scores: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._cumulative: np.ndarray | None = None

    def calibrate(
        self, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
    ) -> WeightedConformal:
        """Store weighted calibration scores."""
        scores = absolute_residual_score(y_true, y_pred)
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != scores.shape:
            raise ValueError(f"weights must have shape {scores.shape}, got {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        if w.sum() <= 0.0:
            raise ValueError("weights must not all be zero")
        order = np.argsort(scores, kind="stable")
        self._scores = scores[order]
        self._weights = w[order]
        self._cumulative = np.cumsum(self._weights)
        return self

    @property
    def n_calibration(self) -> int:
        """Number of calibration scores."""
        if self._scores is None:
            raise RuntimeError("call calibrate(...) first")
        return int(self._scores.size)

    @property
    def weight_sum(self) -> float:
        """``sum_i w(x_i)`` [-]."""
        if self._weights is None:
            raise RuntimeError("call calibrate(...) first")
        return float(self._weights.sum())

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size of the calibration weights [-]."""
        from .bounds import effective_sample_size as _ess

        if self._weights is None:
            raise RuntimeError("call calibrate(...) first")
        return _ess(self._weights)

    def quantiles(self, test_weights: np.ndarray) -> np.ndarray:
        """Per-test-point conformal half-widths [Wh], ``inf`` where unbounded."""
        if self._scores is None or self._cumulative is None or self._weights is None:
            raise RuntimeError("call calibrate(...) first")
        wt = np.asarray(test_weights, dtype=float).ravel()
        if np.any(wt < 0.0):
            raise ValueError("test_weights must be non-negative")
        if not np.all(np.isfinite(wt)):
            raise ValueError("test_weights must be finite")
        total = self.weight_sum + wt
        threshold = (1.0 - self.alpha) * total * (1.0 - QUANTILE_RELATIVE_TOLERANCE)
        index = np.searchsorted(self._cumulative, threshold, side="left")
        out = np.full(wt.shape, np.inf, dtype=float)
        inside = index < self._scores.size
        out[inside] = self._scores[index[inside]]
        return out

    def interval(self, y_pred: np.ndarray, test_weights: np.ndarray) -> Interval:
        """Symmetric per-point interval [Wh]."""
        p = np.asarray(y_pred, dtype=float).ravel()
        wt = np.asarray(test_weights, dtype=float).ravel()
        if p.shape != wt.shape:
            raise ValueError(f"y_pred and test_weights must agree, got {p.shape} and {wt.shape}")
        q = self.quantiles(wt)
        return Interval(lower=p - q, point=p, upper=p + q)


class ConformalizedRegressor:
    """A fitted point predictor plus a calibrated conformal band.

    This is the object that turns a point-estimating regressor into one with
    an uncertainty output, which is the only form in which a learned
    component in this package is allowed to be used.

    Parameters
    ----------
    predictor:
        Anything with ``predict(features) -> ndarray``.
    calibrator:
        A calibrated :class:`SplitConformal`.
    """

    def __init__(self, predictor: Predictor, calibrator: SplitConformal) -> None:
        if not calibrator.is_calibrated:
            raise ValueError("calibrator must be calibrated before wrapping")
        self.predictor = predictor
        self.calibrator = calibrator

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Point prediction [Wh]."""
        return np.asarray(self.predictor.predict(features), dtype=float).ravel()

    def predict_interval(self, features: np.ndarray) -> Interval:
        """Point prediction with its conformal interval [Wh]."""
        return self.calibrator.interval(self.predict(features))
