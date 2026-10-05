"""Multivariate novelty models, each with a calibrated confidence output.

Three models are provided and all three are *one-class*: they are fitted on
nominal data only and never see a labelled anomaly, because an operational
monitor has plenty of nominal telemetry and no catalogue of the faults it has
not had yet.

:class:`HotellingT2Q`
    The classical multivariate statistical-process-control baseline, and the
    thing the machine-learning models have to beat.  PCA on the nominal data,
    Hotelling's ``T^2`` on the retained principal components, and the squared
    prediction error ``Q`` (also called SPE) on the residual subspace.  Each
    statistic is divided by its own nominal upper quantile so the two are
    commensurable, and the score is the larger of the two ratios.
:class:`GmmNovelty`
    Negative log density under a Gaussian mixture fitted to the nominal data.
:class:`IsolationForestNovelty`
    Negative :meth:`sklearn.ensemble.IsolationForest.score_samples`.

Confidence output
-----------------
Every model exposes :meth:`NoveltyModel.novelty_pvalue`, the empirical
probability that a nominal sample would score at least as high as the sample
presented, with a Clopper-Pearson interval reflecting the finite size of the
calibration set.  The interval width is a property of the calibration sample,
not of the sample being scored, and it is the honest statement of how finely
the tail is resolved: with ``n`` calibration samples no p-value below ``1 / n``
is distinguishable from zero, and
:meth:`NoveltyModel.pvalue_resolution_floor` reports that bound.

References
----------
Hotelling, H. (1947). "Multivariate Quality Control." In Eisenhart, Hastay and
    Wallis (eds.), *Techniques of Statistical Analysis*, McGraw-Hill, 111-184.
Jackson, J. E. and Mudholkar, G. S. (1979). "Control Procedures for Residuals
    Associated with Principal Component Analysis." *Technometrics* 21(3),
    341-349.  Origin of the ``Q`` / SPE residual statistic used here.
Liu, F. T., Ting, K. M. and Zhou, Z.-H. (2008). "Isolation Forest."
    *Proceedings of the 8th IEEE International Conference on Data Mining*,
    413-422.
Clopper, C. J. and Pearson, E. S. (1934). "The Use of Confidence or Fiducial
    Limits Illustrated in the Case of the Binomial." *Biometrika* 26(4),
    404-413.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from .calibration import clopper_pearson_interval

__all__ = [
    "PValue",
    "NoveltyModel",
    "HotellingT2Q",
    "GmmNovelty",
    "IsolationForestNovelty",
]


@dataclass(frozen=True)
class PValue:
    """An empirical tail probability with its finite-calibration uncertainty.

    Attributes
    ----------
    pvalue
        ``(1 + #{nominal calibration scores >= score}) / (1 + n_calibration)``,
        the standard conservative plug-in estimate, dimensionless in (0, 1].
    low, high
        Clopper-Pearson interval at ``confidence`` on the underlying exceedance
        probability.
    confidence
        Nominal coverage of the interval.
    n_calibration
        Calibration sample size.
    resolution_floor
        ``1 / (1 + n_calibration)``: the smallest p-value the calibration set
        can express.  A reported ``pvalue`` equal to this is a statement that
        the score was not exceeded by any calibration sample, not that the
        probability is that small.
    """

    pvalue: float
    low: float
    high: float
    confidence: float
    n_calibration: int
    resolution_floor: float


class NoveltyModel(ABC):
    """Base class: fit on nominal data, score, and report a calibrated p-value.

    Subclasses implement :meth:`_fit` and :meth:`_raw_score`.  The base class
    standardises the features (mean and standard deviation from the fit data),
    stores the sorted nominal score vector for the p-value, and validates
    shapes.
    """

    name: str = "novelty"

    def __init__(self) -> None:
        self._mean: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None
        self._cal_scores: NDArray[np.float64] | None = None

    # -- subclass hooks ---------------------------------------------------- #

    @abstractmethod
    def _fit(self, z: NDArray[np.float64]) -> None:
        """Fit on standardised features ``z`` of shape ``(n_samples, n_features)``."""

    @abstractmethod
    def _raw_score(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Novelty score for standardised features; higher means more novel."""

    # -- public API -------------------------------------------------------- #

    @property
    def n_features(self) -> int:
        """Number of features the model was fitted on."""
        self._require_fitted()
        assert self._mean is not None
        return int(self._mean.size)

    def fit(self, x: NDArray[np.float64]) -> NoveltyModel:
        """Fit on nominal samples ``x`` of shape ``(n_samples, n_features)``.

        Feature standardisation uses the mean and the population standard
        deviation of ``x``; a feature with zero variance raises ``ValueError``
        rather than being silently divided by a floor.
        """
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"x must be 2-D (n_samples, n_features), got {arr.shape}")
        if arr.shape[0] < 2:
            raise ValueError(f"need at least 2 samples to fit, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x contains non-finite values")
        mean = arr.mean(axis=0)
        scale = arr.std(axis=0)
        bad = np.flatnonzero(scale <= 0.0)
        if bad.size:
            raise ValueError(f"features {bad.tolist()} have zero variance in the fit data")
        self._mean, self._scale = mean, scale
        z = (arr - mean) / scale
        self._fit(z)
        self._cal_scores = np.sort(self._raw_score(z))
        return self

    def score(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Novelty score for ``x`` of shape ``(n_samples, n_features)``.

        Returns shape ``(n_samples,)``; higher means more novel.  Units are
        model-specific and only orderings and calibrated thresholds are
        meaningful.
        """
        self._require_fitted()
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {arr.shape}")
        if arr.shape[1] != self.n_features:
            raise ValueError(
                f"x has {arr.shape[1]} features, model was fitted on {self.n_features}"
            )
        assert self._mean is not None and self._scale is not None
        return self._raw_score((arr - self._mean) / self._scale)

    def novelty_pvalue(
        self, x: NDArray[np.float64], confidence: float = 0.95
    ) -> list[PValue]:
        """Calibrated tail p-value with a Clopper-Pearson interval, one per sample.

        Parameters
        ----------
        x
            Shape ``(n_samples, n_features)``.
        confidence
            Nominal coverage of the interval, in (0, 1).

        Returns
        -------
        list of PValue
        """
        self._require_fitted()
        assert self._cal_scores is not None
        s = self.score(x)
        n = int(self._cal_scores.size)
        # #{cal >= score} using the sorted calibration vector.
        exceed = n - np.searchsorted(self._cal_scores, s, side="left")
        out = []
        for k in exceed.tolist():
            lo, hi = clopper_pearson_interval(k, n, confidence)
            out.append(
                PValue(
                    pvalue=(1 + k) / (1 + n),
                    low=lo,
                    high=hi,
                    confidence=float(confidence),
                    n_calibration=n,
                    resolution_floor=1.0 / (1 + n),
                )
            )
        return out

    def pvalue_resolution_floor(self) -> float:
        """Smallest p-value the calibration set can express, ``1 / (1 + n_cal)``."""
        self._require_fitted()
        assert self._cal_scores is not None
        return 1.0 / (1 + int(self._cal_scores.size))

    def _require_fitted(self) -> None:
        if self._mean is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")


class HotellingT2Q(NoveltyModel):
    """PCA-based ``T^2`` / ``Q`` monitor (Hotelling 1947; Jackson & Mudholkar 1979).

    Parameters
    ----------
    n_components
        Number of principal components retained in the ``T^2`` subspace.
        ``None`` retains the smallest number explaining at least
        ``variance_target`` of the nominal variance.
    variance_target
        Used only when ``n_components`` is ``None``; in (0, 1).
    inner_quantile
        Nominal quantile at which each of ``T^2`` and ``Q`` is normalised before
        the two are combined with ``max``, in (0, 1).  This only sets the
        relative weighting of the two statistics; the operating point itself is
        set later by :func:`telemetryool.calibration.calibrate_threshold`.

    Notes
    -----
    The ``T^2`` statistic is ``sum_j t_j^2 / lambda_j`` over the retained
    components, with ``lambda_j`` the nominal eigenvalues, and ``Q`` is the
    squared norm of the residual after reconstruction from those components.
    Both are computed on the standardised features, so this is PCA on the
    correlation matrix.  Validity: the usual ``T^2`` reference distribution
    assumes multivariate normal nominal data; that assumption is **not** used
    here, because the quantiles are taken empirically from the fit data.
    """

    name = "t2q"

    def __init__(
        self,
        n_components: int | None = None,
        variance_target: float = 0.9,
        inner_quantile: float = 0.99,
    ) -> None:
        super().__init__()
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1 or None, got {n_components!r}")
        if not (0.0 < float(variance_target) < 1.0):
            raise ValueError(f"variance_target must lie in (0, 1), got {variance_target!r}")
        if not (0.0 < float(inner_quantile) < 1.0):
            raise ValueError(f"inner_quantile must lie in (0, 1), got {inner_quantile!r}")
        self.n_components = n_components
        self.variance_target = float(variance_target)
        self.inner_quantile = float(inner_quantile)
        self._vectors: NDArray[np.float64] | None = None
        self._eigenvalues: NDArray[np.float64] | None = None
        self._t2_ref = 1.0
        self._q_ref = 1.0
        self._kept = 0

    @property
    def n_retained(self) -> int:
        """Number of principal components retained in the ``T^2`` subspace."""
        return self._kept

    def _fit(self, z: NDArray[np.float64]) -> None:
        cov = np.cov(z, rowvar=False)
        cov = np.atleast_2d(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        vals = np.clip(vals, 1e-12, None)
        if self.n_components is None:
            frac = np.cumsum(vals) / vals.sum()
            kept = int(np.searchsorted(frac, self.variance_target) + 1)
        else:
            kept = min(int(self.n_components), vals.size)
        self._kept = max(1, min(kept, vals.size))
        self._vectors, self._eigenvalues = vecs, vals
        t2, q = self._parts(z)
        self._t2_ref = float(max(np.quantile(t2, self.inner_quantile), 1e-12))
        self._q_ref = float(max(np.quantile(q, self.inner_quantile), 1e-12))

    def _parts(
        self, z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self._vectors is not None and self._eigenvalues is not None
        kept = self._kept
        scores = z @ self._vectors
        t2 = (scores[:, :kept] ** 2 / self._eigenvalues[:kept]).sum(axis=1)
        if kept >= self._eigenvalues.size:
            q = np.zeros(z.shape[0])
        else:
            q = (scores[:, kept:] ** 2).sum(axis=1)
        return t2, q

    def statistics(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Raw ``(T^2, Q)`` for ``x``, shape ``(n_samples,)`` each, dimensionless."""
        self._require_fitted()
        assert self._mean is not None and self._scale is not None
        return self._parts((np.asarray(x, dtype=float) - self._mean) / self._scale)

    def _raw_score(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        t2, q = self._parts(z)
        return np.maximum(t2 / self._t2_ref, q / self._q_ref)


class GmmNovelty(NoveltyModel):
    """Negative log density under a Gaussian mixture fitted to nominal data.

    Parameters
    ----------
    n_components
        Mixture components, >= 1.
    covariance_type
        Passed to :class:`sklearn.mixture.GaussianMixture`.
    random_state
        Seed for the EM initialisation; fixed so a run is reproducible.
    max_iter
        EM iteration cap.

    Notes
    -----
    Fitting is scikit-learn EM with k-means initialisation.  The score is
    ``-log p(x)``, in nats, which is unbounded above and therefore has no
    natural threshold -- the threshold is supplied by calibration.
    """

    name = "gmm"

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        random_state: int = 0,
        max_iter: int = 100,
    ) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components!r}")
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self._model: GaussianMixture | None = None

    def _fit(self, z: NDArray[np.float64]) -> None:
        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
        ).fit(z)

    def _raw_score(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._model is not None
        return -self._model.score_samples(z)


class IsolationForestNovelty(NoveltyModel):
    """Isolation Forest novelty score (Liu, Ting & Zhou 2008).

    Parameters
    ----------
    n_estimators
        Number of trees.  The default of 60 is a compute-budget choice, not a
        tuned optimum; see the README compute budget.
    max_samples
        Sub-sample size per tree.
    random_state
        Seed, fixed for reproducibility.
    """

    name = "iforest"

    def __init__(
        self,
        n_estimators: int = 60,
        max_samples: int = 256,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators!r}")
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.random_state = int(random_state)
        self._model: IsolationForest | None = None

    def _fit(self, z: NDArray[np.float64]) -> None:
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, z.shape[0]),
            random_state=self.random_state,
        ).fit(z)

    def _raw_score(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._model is not None
        return -self._model.score_samples(z)
