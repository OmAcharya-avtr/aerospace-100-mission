r"""Classical baselines and the learned multi-sensor estimator.

Order of construction, per the mission's rule for AI products: the closed-form
inversions of :mod:`turbscope.inversion` were implemented and validated **first**;
this module wraps them as scoreable baselines and then adds a learned model that
is benchmarked against them on the same held-out rows.

Baselines (no fitting, no data)
-------------------------------
``WeakScintillometerBaseline``
    The textbook linear inversion ``beta_0^2 = sigma_I^2`` applied to the point
    channel.  Correct to first order for ``beta_0^2 < 0.3`` and increasingly
    wrong above it.
``SaturationAwareBaseline``
    Root-finds the Andrews-Phillips index on the point channel and takes the
    lowest branch.  Above the attainable maximum it returns the value at the peak,
    which is the least-wrong answer the model can give, and flags the row.
``ApertureChannelBaseline``
    The same root-find on the aperture-averaged channel, whose peak sits at a much
    larger ``beta_0^2``: aperture averaging is the classical way to postpone
    saturation (Wang, Ochs & Clifford 1978, *JOSA* 68(3), 334-338, introduced the
    large-aperture scintillometer for exactly this reason -- note that their
    double-aperture geometry is *not* what is modelled here, only the
    single-ended receiver-aperture averaging of Andrews & Phillips 2005 Eq. 9.60).
``DimmBaseline``
    Sarazin & Roddier (1990) plus Fried (1966).  It estimates the *coherence*
    weighted path average, so on a non-uniform path it is biased with respect to
    the scintillation-weighted target by construction; that bias is reported, not
    hidden.

Learned model
-------------
Three ``sklearn.ensemble.GradientBoostingRegressor`` quantile fits
(alpha = 0.05 / 0.50 / 0.95) on the 13 features of
:data:`turbscope.dataset.FEATURE_NAMES`, plus a split-conformal offset fitted on
a disjoint calibration split (Romano, Patterson & Candes 2019, *Conformalized
Quantile Regression*, NeurIPS 32).  Target: ``log10`` of the
scintillation-kernel weighted path average of ``Cn2``.

**No uncertainty claim here transfers to a real instrument.**  The conformal
guarantee needs calibration and deployment data to be exchangeable, and here both
come from the same synthetic generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ._validate import check_count, check_probability
from .dataset import FEATURE_NAMES, SyntheticDataset, generate_dataset
from .geometry import PathGeometry
from .inversion import scintillation_branches
from .scintillation import saturation_peak, uniform_cn2_from_beta0_sq

__all__ = [
    "ApertureChannelBaseline",
    "BASELINES",
    "DimmBaseline",
    "Prediction",
    "SaturationAwareBaseline",
    "TurbScopeModel",
    "WeakScintillometerBaseline",
    "split_dataset",
    "train_default_model",
]

_CN2_FLOOR = 1e-20


class _Baseline:
    """Common interface: ``predict(dataset) -> log10 Cn2`` array."""

    name = "baseline"

    def predict(self, data: SyntheticDataset) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class WeakScintillometerBaseline(_Baseline):
    """Textbook weak-fluctuation inversion of the point scintillometer channel."""

    name = "weak closed form (point scintillometer)"

    def predict(self, data: SyntheticDataset) -> np.ndarray:
        return data.x[:, FEATURE_NAMES.index("log10_cn2_weak_point")].copy()


class _RootFindBaseline(_Baseline):
    """Shared root-finding inversion of the Andrews-Phillips index."""

    def _channel(self, data: SyntheticDataset) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict(self, data: SyntheticDataset) -> np.ndarray:
        sigma, d_sq = self._channel(data)
        out = np.empty(sigma.size, dtype=float)
        peak_cache: dict[float, tuple[float, float]] = {}
        for i in range(sigma.size):
            d2 = float(d_sq[i])
            s = float(max(sigma[i], 1e-12))
            key = round(d2, 9)
            if key not in peak_cache:
                peak_cache[key] = saturation_peak(d2)
            b_peak, s_peak = peak_cache[key]
            if s >= s_peak:
                beta = b_peak
            else:
                branches = scintillation_branches(s, d2)
                beta = branches[0] if branches else b_peak
            path = PathGeometry(float(data.path_length_m[i]), float(data.wavelength_m[i]))
            out[i] = np.log10(max(uniform_cn2_from_beta0_sq(beta, path), _CN2_FLOOR))
        return out


class SaturationAwareBaseline(_RootFindBaseline):
    """Root-find the point-channel index; lowest branch, clipped at the peak."""

    name = "saturation-aware inversion (point scintillometer)"

    def _channel(self, data: SyntheticDataset) -> tuple[np.ndarray, np.ndarray]:
        return data.sigma_i2_point, np.zeros_like(data.sigma_i2_point)


class ApertureChannelBaseline(_RootFindBaseline):
    """Root-find the aperture-averaged channel; lowest branch, clipped at the peak."""

    name = "saturation-aware inversion (aperture-averaged channel)"

    def _channel(self, data: SyntheticDataset) -> tuple[np.ndarray, np.ndarray]:
        return data.sigma_i2_aperture, data.aperture_d_sq


class DimmBaseline(_Baseline):
    """Sarazin & Roddier (1990) + Fried (1966) DIMM inversion (coherence kernel)."""

    name = "DIMM closed form (coherence kernel)"

    def predict(self, data: SyntheticDataset) -> np.ndarray:
        return data.x[:, FEATURE_NAMES.index("log10_cn2_dimm")].copy()


BASELINES: tuple[_Baseline, ...] = (
    WeakScintillometerBaseline(),
    SaturationAwareBaseline(),
    ApertureChannelBaseline(),
    DimmBaseline(),
)


@dataclass(frozen=True)
class Prediction:
    """Learned-model output: point estimate plus a prediction interval."""

    cn2: np.ndarray
    cn2_lower: np.ndarray
    cn2_upper: np.ndarray
    log10_cn2: np.ndarray
    log10_lower: np.ndarray
    log10_upper: np.ndarray
    coverage: float
    extrapolating: np.ndarray


class TurbScopeModel:
    """Quantile gradient boosting with a split-conformal interval.

    Parameters
    ----------
    coverage
        Nominal interval coverage; only 0.90 has been calibrated and measured.
    n_estimators, max_depth, learning_rate, min_samples_leaf, random_state
        Passed to :class:`sklearn.ensemble.GradientBoostingRegressor`.
    """

    def __init__(
        self,
        coverage: float = 0.90,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.06,
        min_samples_leaf: int = 20,
        random_state: int = 13,
    ) -> None:
        self.coverage = check_probability("coverage", coverage)
        alpha = 1.0 - self.coverage
        common = dict(
            loss="quantile",
            n_estimators=check_count("n_estimators", n_estimators),
            max_depth=check_count("max_depth", max_depth),
            learning_rate=float(learning_rate),
            min_samples_leaf=check_count("min_samples_leaf", min_samples_leaf),
            random_state=int(random_state),
        )
        self._median = GradientBoostingRegressor(alpha=0.5, **common)
        self._lower = GradientBoostingRegressor(alpha=alpha / 2.0, **common)
        self._upper = GradientBoostingRegressor(alpha=1.0 - alpha / 2.0, **common)
        self._delta = 0.0
        self._fitted = False
        self._x_min: np.ndarray | None = None
        self._x_max: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> TurbScopeModel:
        """Fit the three quantile regressors on ``(x, y=log10 Cn2)``."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"x must have shape (n, {len(FEATURE_NAMES)}) matching FEATURE_NAMES, "
                f"got {x.shape}"
            )
        if y.shape != (x.shape[0],):
            raise ValueError(f"y must have shape ({x.shape[0]},), got {y.shape}")
        for m in (self._median, self._lower, self._upper):
            m.fit(x, y)
        self._x_min = x.min(axis=0)
        self._x_max = x.max(axis=0)
        self._delta = 0.0
        self._fitted = True
        return self

    def calibrate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fit the split-conformal offset on a **disjoint** calibration set.

        Conformity score ``E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))``; the offset
        is its ``ceil((n+1)(1-alpha))``-th order statistic (Romano et al. 2019).
        Returns the offset in dex.
        """
        self._require_fit()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lo = self._lower.predict(x)
        hi = self._upper.predict(x)
        scores = np.maximum(lo - y, y - hi)
        n = scores.size
        rank = int(np.ceil((n + 1) * self.coverage))
        rank = min(max(rank, 1), n)
        self._delta = float(np.sort(scores)[rank - 1])
        return self._delta

    @property
    def conformal_offset(self) -> float:
        """Current split-conformal offset, dex."""
        return self._delta

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() first")

    def predict_log10(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lower, median, upper)`` in ``log10 Cn2``, sorted pointwise."""
        self._require_fit()
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"x must have {len(FEATURE_NAMES)} columns matching FEATURE_NAMES, "
                f"got {x.shape[1]}"
            )
        med = self._median.predict(x)
        lo = self._lower.predict(x) - self._delta
        hi = self._upper.predict(x) + self._delta
        stack = np.sort(np.vstack([lo, med, hi]), axis=0)
        return stack[0], stack[1], stack[2]

    def predict(self, x: np.ndarray) -> Prediction:
        """Full prediction with linear-space bounds and an extrapolation flag."""
        lo, med, hi = self.predict_log10(x)
        x2 = np.atleast_2d(np.asarray(x, dtype=float))
        assert self._x_min is not None and self._x_max is not None
        extrap = np.any((x2 < self._x_min) | (x2 > self._x_max), axis=1)
        return Prediction(
            cn2=10.0**med,
            cn2_lower=10.0**lo,
            cn2_upper=10.0**hi,
            log10_cn2=med,
            log10_lower=lo,
            log10_upper=hi,
            coverage=self.coverage,
            extrapolating=extrap,
        )


def split_dataset(
    n: int, seed: int = 4321, fractions: tuple[float, float, float] = (0.55, 0.20, 0.25)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random disjoint fit / calibration / test index arrays.

    Every scenario is an independent draw (independent path, instrument and noise),
    so a row-level split is a scenario-level split here -- there are no repeated
    measurements of one path to leak across the boundary.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(check_count("n", n, minimum=10))
    n_fit = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    return perm[:n_fit], perm[n_fit : n_fit + n_cal], perm[n_fit + n_cal :]


def train_default_model(
    n_scenarios: int = 6000,
    data_seed: int = 20260829,
    split_seed: int = 4321,
    coverage: float = 0.90,
    calibrate: bool = True,
) -> tuple[TurbScopeModel, dict[str, object]]:
    """Generate data, fit and conformalise the default model.

    Returns the model and a dictionary of artefacts: the dataset and the three
    index arrays, so that a caller can reproduce every reported metric.
    """
    data = generate_dataset(n_scenarios, seed=data_seed)
    idx_fit, idx_cal, idx_test = split_dataset(len(data), seed=split_seed)
    model = TurbScopeModel(coverage=coverage)
    model.fit(data.x[idx_fit], data.y[idx_fit])
    if calibrate:
        model.calibrate(data.x[idx_cal], data.y[idx_cal])
    return model, {
        "data": data,
        "idx_fit": idx_fit,
        "idx_cal": idx_cal,
        "idx_test": idx_test,
        "conformal_offset": model.conformal_offset,
    }
