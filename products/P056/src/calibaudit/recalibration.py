"""Recalibration maps, and a held-out audit of whether they helped.

Three forecasters are compared on the same held-out split:

``RawForecast``
    the identity map. **This is the baseline and it is not a straw man.** A
    forecast that is already calibrated cannot be improved by recalibration,
    and any map fitted on a finite training split adds estimation variance,
    so the raw forecast wins whenever the miscalibration it could remove is
    smaller than the noise the fit introduces.

``PlattScaling``
    a two-parameter logistic map ``sigmoid(a * logit(f) + b)`` fitted by
    maximum likelihood (Platt 1999). The learned component with the least
    capacity: it can rescale and shift confidence but cannot change the
    ranking of forecasts, so it cannot repair a non-monotone calibration
    curve.

``IsotonicCalibration``
    a monotone step function fitted by pool-adjacent-violators (Zadrozny and
    Elkan 2002), through ``sklearn.isotonic.IsotonicRegression``. Nonparametric
    and therefore the first to overfit a small training split.

Both fitted maps expose an uncertainty output, not only a point estimate:
``predict_with_interval`` returns a pointwise percentile interval from a
bootstrap ensemble of maps fitted at ``fit`` time, so a user can see where the
recalibration itself is poorly determined. On a small training split those
intervals are wide, which is the same fact as the honest negative above seen
from the other side.

A note on scikit-learn 1.9: ``CalibratedClassifierCV(estimator, cv="prefit")``
no longer exists. ``cv`` must be an int, a splitter, an iterable or ``None``,
and the prefit path is now spelled
``CalibratedClassifierCV(FrozenEstimator(estimator))``. The exact exception is
recorded in ``validation/outputs/validate_sklearn_interop_output.txt``. This
package fits maps directly on forecast values and never takes that path, but
anyone recalibrating a prefit classifier on this version will.

References
----------
Platt, J. C. (1999). "Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods." In *Advances in Large Margin
Classifiers*, MIT Press, 61-74.

Zadrozny, B. and Elkan, C. (2002). "Transforming classifier scores into
accurate multiclass probability estimates." *KDD 2002*, 694-699.
doi:10.1145/775047.775151

Niculescu-Mizil, A. and Caruana, R. (2005). "Predicting good probabilities
with supervised learning." *ICML 2005*, 625-632. doi:10.1145/1102351.1102430
(reports isotonic regression needing of order 1000 training samples before it
beats Platt scaling, which is the effect measured here)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from .ece import expected_calibration_error
from .scores import brier_score, check_forecasts, log_score, skill_score

__all__ = [
    "LOGIT_CLIP",
    "METHOD_NAMES",
    "IsotonicCalibration",
    "MethodResult",
    "PlattScaling",
    "RawForecast",
    "RecalibrationAudit",
    "SampleSizeRow",
    "SampleSizeSweep",
    "get_method",
    "recalibration_audit",
    "sample_size_sweep",
]

#: Forecasts are clipped into ``[LOGIT_CLIP, 1 - LOGIT_CLIP]`` before the logit,
#: so that a forecast of exactly 0 or 1 gives a large finite logit instead of
#: an infinity. The value bounds the logit at +/- 27.6.
LOGIT_CLIP = 1e-12

#: Names accepted by :func:`get_method`, baseline first.
METHOD_NAMES: tuple[str, ...] = ("raw", "platt", "isotonic")


def _safe_logit(f: np.ndarray) -> np.ndarray:
    return logit(np.clip(f, LOGIT_CLIP, 1.0 - LOGIT_CLIP))


class Recalibrator(ABC):
    """A map from forecast to recalibrated forecast, fitted on held-in data."""

    name: str = "abstract"
    is_learned: bool = True

    def __init__(self, *, n_bootstrap: int = 0, seed: int = 0) -> None:
        if n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be non-negative, got {n_bootstrap!r}")
        self.n_bootstrap = int(n_bootstrap)
        self.seed = int(seed)
        self._fitted = False
        self._ensemble: list[Recalibrator] = []

    @abstractmethod
    def _fit(self, f: np.ndarray, o: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, f: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _clone(self, seed: int) -> Recalibrator: ...

    def fit(self, forecasts: ArrayLike, outcomes: ArrayLike) -> Recalibrator:
        """Fit on the training split; returns ``self``."""
        f, o = check_forecasts(forecasts, outcomes, min_samples=2)
        self._fit(f, o)
        self._fitted = True
        self._ensemble = []
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.seed)
            for b in range(self.n_bootstrap):
                idx = rng.integers(0, f.size, size=f.size)
                member = self._clone(self.seed * 31 + b + 1)
                try:
                    member._fit(f[idx], o[idx])
                except (ValueError, RuntimeError):
                    continue
                member._fitted = True
                self._ensemble.append(member)
        return self

    def predict(self, forecasts: ArrayLike) -> np.ndarray:
        """Recalibrated probabilities, same shape as ``forecasts``."""
        if not self._fitted:
            raise ValueError(f"{type(self).__name__} is not fitted; call fit() first")
        f = np.asarray(forecasts, dtype=float)
        f, _ = check_forecasts(f, np.zeros_like(f))
        return np.clip(self._predict(f), 0.0, 1.0)

    def predict_with_interval(
        self, forecasts: ArrayLike, *, level: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(point, lower, upper)`` from the bootstrap ensemble.

        The interval covers the uncertainty of the *fitted map* at each
        forecast value, not the uncertainty of the outcome. It is the width a
        user should look at before trusting a recalibrated probability.

        Raises
        ------
        ValueError
            If the recalibrator was constructed with ``n_bootstrap = 0``, so
            no ensemble exists, or if ``level`` is outside (0, 1).
        """
        point = self.predict(forecasts)
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}")
        if not self._ensemble:
            raise ValueError(
                f"{type(self).__name__} has no bootstrap ensemble; construct it with "
                "n_bootstrap > 0 to obtain intervals"
            )
        draws = np.vstack([m.predict(forecasts) for m in self._ensemble])
        alpha = (1.0 - float(level)) / 2.0
        return point, np.quantile(draws, alpha, axis=0), np.quantile(draws, 1 - alpha, axis=0)

    @property
    def n_ensemble(self) -> int:
        """Number of bootstrap members that converged."""
        return len(self._ensemble)


class RawForecast(Recalibrator):
    """The baseline: the identity map, fitted on nothing."""

    name = "raw"
    is_learned = False

    def _fit(self, f: np.ndarray, o: np.ndarray) -> None:
        del f, o

    def _predict(self, f: np.ndarray) -> np.ndarray:
        return f.copy()

    def _clone(self, seed: int) -> RawForecast:
        return RawForecast(n_bootstrap=0, seed=seed)


class PlattScaling(Recalibrator):
    """``sigmoid(a * logit(f) + b)``, fitted by maximum likelihood.

    Parameters
    ----------
    target_smoothing
        When True, use Platt's (1999) smoothed targets
        ``t+ = (N+ + 1) / (N+ + 2)``, ``t- = 1 / (N- + 2)`` instead of 1 and 0.
        This is a shrinkage that keeps the fit finite when the two classes are
        separable in ``logit(f)``; it is off by default so that the plain
        maximum-likelihood fit is what gets measured.
    n_bootstrap, seed
        Size and seed of the bootstrap ensemble used by
        :meth:`predict_with_interval`. 0 disables it.

    Attributes
    ----------
    a_, b_
        Fitted slope (dimensionless) and intercept (in logit units). ``a_ = 1``
        and ``b_ = 0`` is the identity map.
    """

    name = "platt"

    def __init__(
        self, *, target_smoothing: bool = False, n_bootstrap: int = 0, seed: int = 0
    ) -> None:
        super().__init__(n_bootstrap=n_bootstrap, seed=seed)
        self.target_smoothing = bool(target_smoothing)
        self.a_: float = 1.0
        self.b_: float = 0.0
        self.optimizer_message_: str = "not fitted"

    def _fit(self, f: np.ndarray, o: np.ndarray) -> None:
        z = _safe_logit(f)
        if self.target_smoothing:
            n_pos = float(np.sum(o == 1.0))
            n_neg = float(np.sum(o == 0.0))
            t = np.where(o == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        else:
            t = o

        def nll_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = float(theta[0]), float(theta[1])
            s = a * z + b
            # log(1 + exp(s)) computed stably
            softplus = np.logaddexp(0.0, s)
            nll = float(np.mean(softplus - t * s))
            p = expit(s)
            r = p - t
            return nll, np.array([float(np.mean(r * z)), float(np.mean(r))])

        res = minimize(
            nll_and_grad,
            x0=np.array([1.0, 0.0]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
        )
        self.a_ = float(res.x[0])
        self.b_ = float(res.x[1])
        self.optimizer_message_ = str(res.message)
        if not res.success and not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"Platt fit did not produce finite parameters: {res.message}")

    def _predict(self, f: np.ndarray) -> np.ndarray:
        return expit(self.a_ * _safe_logit(f) + self.b_)

    def _clone(self, seed: int) -> PlattScaling:
        return PlattScaling(
            target_smoothing=self.target_smoothing, n_bootstrap=0, seed=seed
        )


class IsotonicCalibration(Recalibrator):
    """Monotone nonparametric map by pool-adjacent-violators.

    Wraps ``sklearn.isotonic.IsotonicRegression`` with ``y_min = 0``,
    ``y_max = 1``, ``increasing = True`` and ``out_of_bounds = "clip"``, so a
    test forecast outside the training range is mapped to the nearest fitted
    endpoint rather than extrapolated.
    """

    name = "isotonic"

    def __init__(self, *, n_bootstrap: int = 0, seed: int = 0) -> None:
        super().__init__(n_bootstrap=n_bootstrap, seed=seed)
        self._model: IsotonicRegression | None = None

    def _fit(self, f: np.ndarray, o: np.ndarray) -> None:
        model = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        model.fit(f, o)
        self._model = model

    def _predict(self, f: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("IsotonicCalibration is not fitted; call fit() first")
        return np.asarray(self._model.predict(f), dtype=float)

    def _clone(self, seed: int) -> IsotonicCalibration:
        return IsotonicCalibration(n_bootstrap=0, seed=seed)

    @property
    def n_steps(self) -> int:
        """Number of distinct values the fitted step function takes."""
        if self._model is None:
            raise ValueError("IsotonicCalibration is not fitted; call fit() first")
        return int(np.unique(self._model.y_thresholds_).size)


def get_method(name: str, *, n_bootstrap: int = 0, seed: int = 0) -> Recalibrator:
    """Construct a recalibrator by name. ``"raw"`` is the baseline.

    Raises
    ------
    KeyError
        If the name is not in :data:`METHOD_NAMES`.
    """
    if name == "raw":
        return RawForecast(n_bootstrap=0, seed=seed)
    if name == "platt":
        return PlattScaling(n_bootstrap=n_bootstrap, seed=seed)
    if name == "isotonic":
        return IsotonicCalibration(n_bootstrap=n_bootstrap, seed=seed)
    raise KeyError(f"unknown method {name!r}; known methods are {METHOD_NAMES}")


@dataclass(frozen=True)
class MethodResult:
    """Held-out scores for one recalibration method. All dimensionless."""

    method: str
    is_learned: bool
    brier: float
    log_score: float
    ece: float
    mce: float
    brier_skill_vs_raw: float
    delta_brier: float
    delta_brier_ci: tuple[float, float]
    delta_brier_worse_fraction: float
    verdict: str
    n_ensemble: int

    @property
    def helped(self) -> bool:
        """True only when the paired bootstrap interval lies entirely below 0."""
        return self.verdict == "improved"


@dataclass(frozen=True)
class RecalibrationAudit:
    """Result of a held-out recalibration audit, baseline first."""

    n_train: int
    n_test: int
    n_bins: int
    strategy: str
    level: float
    n_bootstrap: int
    seed: int
    results: tuple[MethodResult, ...]

    def by_method(self, name: str) -> MethodResult:
        for r in self.results:
            if r.method == name:
                return r
        raise KeyError(f"method {name!r} not in this audit")

    def table(self) -> str:
        """Fixed-width table, baseline row first."""
        head = (
            f"{'method':>10} {'learned':>8} {'brier':>10} {'log':>9} {'ECE':>9} "
            f"{'MCE':>9} {'dBrier':>10} {'ci_lo':>10} {'ci_hi':>10} {'verdict':>18}"
        )
        lines = [head, "-" * len(head)]
        for r in self.results:
            lines.append(
                f"{r.method:>10} {str(r.is_learned):>8} {r.brier:>10.6f} "
                f"{r.log_score:>9.6f} {r.ece:>9.6f} {r.mce:>9.6f} "
                f"{r.delta_brier:>+10.6f} {r.delta_brier_ci[0]:>+10.6f} "
                f"{r.delta_brier_ci[1]:>+10.6f} {r.verdict:>18}"
            )
        return "\n".join(lines)


def recalibration_audit(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    methods: tuple[str, ...] = METHOD_NAMES,
    test_fraction: float = 0.5,
    n_bins: int = 10,
    strategy: str = "equal_mass",
    n_bootstrap: int = 400,
    level: float = 0.9,
    seed: int = 0,
    ensemble_size: int = 0,
) -> RecalibrationAudit:
    """Fit each method on a training split and score it on the held-out split.

    The split is a single seeded random permutation, not cross-validation: the
    question being answered is what one practitioner with one dataset gets,
    which is the situation in which recalibration is actually applied.

    ``delta_brier`` is ``BS_method - BS_raw`` on the **same** held-out samples,
    so the comparison is paired; its interval is a paired percentile bootstrap
    over the test set, resampling test indices and recomputing both scores on
    each resample. ``verdict`` is ``"improved"`` when the whole interval is
    below 0, ``"worse"`` when it is entirely above 0, and
    ``"indistinguishable"`` otherwise.

    Raises
    ------
    ValueError
        If ``test_fraction`` leaves fewer than 2 samples on either side, or if
        a requested method name is unknown, or if ``"raw"`` is not among the
        methods (the baseline is not optional).
    """
    f, o = check_forecasts(forecasts, outcomes, min_samples=4)
    if "raw" not in methods:
        raise ValueError("the 'raw' baseline must be included in methods")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1), got {test_fraction!r}")
    n = f.size
    n_test = int(round(test_fraction * n))
    n_train = n - n_test
    if n_test < 2 or n_train < 2:
        raise ValueError(
            f"test_fraction {test_fraction!r} on {n} samples leaves "
            f"{n_train} train / {n_test} test; both must be at least 2"
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    tr, te = perm[:n_train], perm[n_train:]
    f_tr, o_tr, f_te, o_te = f[tr], o[tr], f[te], o[te]

    preds: dict[str, np.ndarray] = {}
    ensembles: dict[str, int] = {}
    for name in methods:
        model = get_method(name, n_bootstrap=int(ensemble_size), seed=int(seed) + 17)
        model.fit(f_tr, o_tr)
        preds[name] = model.predict(f_te)
        ensembles[name] = model.n_ensemble

    boot_idx = rng.integers(0, n_test, size=(int(n_bootstrap), n_test))
    raw_pred = preds["raw"]
    raw_brier = brier_score(raw_pred, o_te)

    results: list[MethodResult] = []
    alpha = (1.0 - float(level)) / 2.0
    for name in methods:
        p = preds[name]
        deltas = np.empty(int(n_bootstrap), dtype=float)
        for b in range(int(n_bootstrap)):
            idx = boot_idx[b]
            d = np.mean((p[idx] - o_te[idx]) ** 2) - np.mean((raw_pred[idx] - o_te[idx]) ** 2)
            deltas[b] = d
        lo = float(np.quantile(deltas, alpha))
        hi = float(np.quantile(deltas, 1.0 - alpha))
        bs = brier_score(p, o_te)
        delta = bs - raw_brier
        if name == "raw":
            verdict = "baseline"
        elif hi < 0.0:
            verdict = "improved"
        elif lo > 0.0:
            verdict = "worse"
        else:
            verdict = "indistinguishable"
        results.append(
            MethodResult(
                method=name,
                is_learned=name != "raw",
                brier=float(bs),
                log_score=log_score(p, o_te),
                ece=expected_calibration_error(p, o_te, n_bins=n_bins, strategy=strategy),
                mce=float(
                    np.max(
                        np.abs(
                            _bin_gaps(p, o_te, n_bins=n_bins, strategy=strategy)
                        )
                    )
                ),
                brier_skill_vs_raw=0.0
                if name == "raw"
                else skill_score(float(bs), float(raw_brier)),
                delta_brier=float(delta),
                delta_brier_ci=(lo, hi),
                delta_brier_worse_fraction=float(np.mean(deltas > 0.0)),
                verdict=verdict,
                n_ensemble=int(ensembles[name]),
            )
        )
    return RecalibrationAudit(
        n_train=int(n_train),
        n_test=int(n_test),
        n_bins=int(n_bins),
        strategy=str(strategy),
        level=float(level),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        results=tuple(results),
    )


def _bin_gaps(
    p: np.ndarray, o: np.ndarray, *, n_bins: int, strategy: str
) -> np.ndarray:
    from .ece import calibration_gaps

    _, gaps = calibration_gaps(p, o, n_bins=n_bins, strategy=strategy)
    return gaps


@dataclass(frozen=True)
class SampleSizeRow:
    """Recalibration outcome at one sample size, averaged over replicates."""

    n_samples: int
    method: str
    mean_brier: float
    mean_delta_brier: float
    sem_delta_brier: float
    harm_rate: float
    improved_rate: float
    n_replicates: int


@dataclass(frozen=True)
class SampleSizeSweep:
    """Recalibration outcome against total sample size."""

    spec_name: str
    test_fraction: float
    rows: tuple[SampleSizeRow, ...] = field(repr=False)

    def table(self) -> str:
        """Fixed-width table, one line per (sample size, method)."""
        head = (
            f"{'n_total':>8} {'method':>10} {'mean_brier':>11} {'mean_dBS':>11} "
            f"{'sem':>10} {'harm_rate':>10} {'improved':>9}"
        )
        lines = [head, "-" * len(head)]
        for r in self.rows:
            lines.append(
                f"{r.n_samples:>8d} {r.method:>10} {r.mean_brier:>11.6f} "
                f"{r.mean_delta_brier:>+11.6f} {r.sem_delta_brier:>10.6f} "
                f"{r.harm_rate:>10.3f} {r.improved_rate:>9.3f}"
            )
        return "\n".join(lines)

    def crossover(self, method: str) -> int | None:
        """Smallest swept ``n_total`` at and above which ``method`` has
        ``mean_delta_brier < 0``, or None if it never does."""
        sizes = sorted({r.n_samples for r in self.rows})
        rows = {(r.n_samples, r.method): r for r in self.rows}
        for i, n in enumerate(sizes):
            if all(rows[(m, method)].mean_delta_brier < 0.0 for m in sizes[i:]):
                return int(n)
        return None


def sample_size_sweep(
    spec,
    *,
    n_samples_grid,
    methods: tuple[str, ...] = METHOD_NAMES,
    n_replicates: int = 60,
    test_fraction: float = 0.5,
    n_bins: int = 10,
    strategy: str = "equal_mass",
    seed: int = 0,
) -> SampleSizeSweep:
    """Does recalibration help, as a function of how much data it was given?

    For each sample size, draw ``n_replicates`` fresh datasets from ``spec``,
    split each, fit each method on the training half and score on the test
    half. ``harm_rate`` is the fraction of replicates in which the method's
    held-out Brier score is strictly worse than the raw forecast's, which is
    the number a practitioner needs and which is rarely published.
    """
    from .synthetic import sample_forecast

    rows: list[SampleSizeRow] = []
    for n_samples in n_samples_grid:
        deltas = {m: np.empty(int(n_replicates)) for m in methods}
        briers = {m: np.empty(int(n_replicates)) for m in methods}
        for r in range(int(n_replicates)):
            s = sample_forecast(spec, int(n_samples), seed=int(seed) * 1000003 + r + 1)
            audit = recalibration_audit(
                s.forecasts,
                s.outcomes,
                methods=methods,
                test_fraction=test_fraction,
                n_bins=n_bins,
                strategy=strategy,
                n_bootstrap=2,
                level=0.9,
                seed=int(seed) * 7 + r + 1,
            )
            for m in methods:
                res = audit.by_method(m)
                deltas[m][r] = res.delta_brier
                briers[m][r] = res.brier
        for m in methods:
            d = deltas[m]
            rows.append(
                SampleSizeRow(
                    n_samples=int(n_samples),
                    method=m,
                    mean_brier=float(briers[m].mean()),
                    mean_delta_brier=float(d.mean()),
                    sem_delta_brier=float(d.std(ddof=1) / np.sqrt(d.size))
                    if d.size > 1
                    else 0.0,
                    harm_rate=float(np.mean(d > 0.0)),
                    improved_rate=float(np.mean(d < 0.0)),
                    n_replicates=int(n_replicates),
                )
            )
    return SampleSizeSweep(
        spec_name=spec.name, test_fraction=float(test_fraction), rows=tuple(rows)
    )
