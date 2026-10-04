"""Link-availability predictors: two deterministic baselines, then a learned model.

Order of construction matters and is recorded here: the climatology baseline
and the logistic-regression baseline were implemented and measured FIRST, and
the learned model is reported against them on identical grouped splits.  If
the climatology baseline is better calibrated, the README and MODEL_CARD say
so and the result stands.

Predictors
----------
1. :class:`ClimatologyBaseline` -- predicts the TRAINING base rate of the
   row's stratum (link type, leg type, elevation band; see
   :func:`~constellink.synthdata.stratify`).  This is the standard
   climatological reference forecast of forecast verification (Wilks 2011,
   "Statistical Methods in the Atmospheric Sciences", 3rd ed., Academic
   Press, Ch. 8): a forecast with no skill beyond knowing what this kind of
   contact usually does.  It is calibrated BY CONSTRUCTION on the training
   distribution, because it reports an observed frequency; its weakness is
   resolution, not reliability.  Unseen strata fall back to the global
   training base rate.
2. :class:`LogisticBaseline` -- L2-regularised logistic regression on
   standardised features (``sklearn.linear_model.LogisticRegression``).  A
   generalised linear model with a canonical link; its fitted probabilities
   are maximum-likelihood under the Bernoulli model, which makes it a
   genuinely competitive calibration reference, not a strawman.
3. :class:`LinkAvailabilityModel` -- a bagged ensemble of
   ``sklearn.ensemble.HistGradientBoostingClassifier`` members, each wrapped
   in ``sklearn.calibration.CalibratedClassifierCV`` with the requested
   calibration method applied on an inner cross-validation split.  The
   ensemble mean is the probability; the ensemble standard deviation is the
   required uncertainty output.

Calibration method
------------------
``method="sigmoid"`` is Platt scaling (Platt 1999, "Probabilistic outputs for
support vector machines...", in Advances in Large Margin Classifiers, MIT
Press); ``method="isotonic"`` is the non-parametric isotonic regression of
Zadrozny & Elkan 2002 ("Transforming classifier scores into accurate
multiclass probability estimates", KDD 2002).  Niculescu-Mizil & Caruana 2005
("Predicting good probabilities with supervised learning", ICML 2005) is the
comparison that established boosted trees need such a correction: their raw
scores are pushed toward the extremes.  Isotonic needs more data than Platt
scaling and overfits on small samples, so ``"sigmoid"`` is the default here
given the dataset size.

Uncertainty output
------------------
The ensemble standard deviation is EPISTEMIC spread only -- disagreement
between members trained on different bootstrap resamples.  It does not
capture the irreducible scintillation fade in the data-generating process
(see :mod:`constellink.synthdata`); the probability itself expresses that.
A near-zero spread at a probability of 0.5 is the correct output for a
contact whose outcome is genuinely a coin flip.

Splitting
---------
:func:`grouped_split` splits by LINK, not by row, so every contact of a given
link pair lands entirely in train or entirely in test.  Row-wise splitting
would leak: contacts of the same link share geometry and station climate, and
a model could memorise the link rather than learn the physics.  The leakage is
quantified in ``validation/validate_calibration.py``, which reports both
splits.

Compute budget: the default configuration (1974 rows, 5 ensemble members,
HistGradientBoosting with 120 iterations and an inner 3-fold calibration) fits
in under 30 s on one core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import (
    brier_decomposition,
    expected_calibration_error,
    log_loss_safe,
)
from .synthdata import N_FEATURES, LinkDataset

__all__ = [
    "ClimatologyBaseline",
    "LogisticBaseline",
    "LinkAvailabilityModel",
    "PredictorScores",
    "score_predictor",
    "grouped_split",
    "row_split",
]


def _check_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != N_FEATURES:
        raise ValueError(f"x must have shape (n, {N_FEATURES}), got {np.shape(x)}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    return x


def _check_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _check_x(x)
    y = np.asarray(y).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"y must have {x.shape[0]} rows, got {y.shape[0]}")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("y must be binary 0/1")
    return x, y.astype(int)


class ClimatologyBaseline:
    """Baseline 1: the training base rate of the row's stratum.

    ``fit`` needs the stratum ids; ``predict_proba`` likewise.  Strata unseen
    in training fall back to the global training base rate, and the count of
    such rows is available as :attr:`n_fallback_last_call`.
    """

    def __init__(self) -> None:
        self._rates: dict[int, float] = {}
        self._global: float = 0.5
        self.n_fallback_last_call: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray, stratum: np.ndarray,
            ) -> ClimatologyBaseline:
        """Record the per-stratum training base rates."""
        x, y = _check_xy(x, y)
        stratum = np.asarray(stratum).ravel()
        if stratum.shape[0] != y.shape[0]:
            raise ValueError("stratum must have one entry per row")
        self._global = float(y.mean())
        self._rates = {int(s): float(y[stratum == s].mean())
                       for s in np.unique(stratum)}
        return self

    def predict_proba(self, x: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        """Return p(close) in [0, 1], shape ``(n,)``."""
        if not self._rates:
            raise RuntimeError("ClimatologyBaseline.fit must be called first")
        x = _check_x(x)
        stratum = np.asarray(stratum).ravel()
        if stratum.shape[0] != x.shape[0]:
            raise ValueError("stratum must have one entry per row")
        self.n_fallback_last_call = int(
            sum(1 for s in stratum if int(s) not in self._rates))
        return np.array([self._rates.get(int(s), self._global) for s in stratum])

    def predict_std(self, x: np.ndarray, stratum: np.ndarray | None = None,
                    ) -> np.ndarray:
        """Uncertainty output: zero, because the climatology is a point estimate.

        ``stratum`` is accepted and ignored so that every predictor exposes the
        same surface.  A zero here states that this predictor has no epistemic
        spread to report; it is not a claim of certainty.
        """
        del stratum
        return np.zeros(_check_x(x).shape[0])


class LogisticBaseline:
    """Baseline 2: L2 logistic regression on standardised features.

    Parameters
    ----------
    c : inverse regularisation strength, > 0.
    max_iter : solver iteration cap, >= 1.
    seed : passed to the solver for determinism.
    """

    def __init__(self, c: float = 1.0, max_iter: int = 2000, seed: int = 0) -> None:
        if c <= 0.0:
            raise ValueError(f"c must be > 0, got {c}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.c = c
        self.max_iter = max_iter
        self.seed = seed
        self._pipe = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticBaseline:
        """Fit the pipeline."""
        x, y = _check_xy(x, y)
        self._pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.c, max_iter=self.max_iter,
                               random_state=self.seed))
        self._pipe.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return p(close) in [0, 1], shape ``(n,)``."""
        if self._pipe is None:
            raise RuntimeError("LogisticBaseline.fit must be called first")
        return self._pipe.predict_proba(_check_x(x))[:, 1]

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Uncertainty output: zero (a single maximum-likelihood fit).

        A parametric confidence interval on the linear predictor could be
        reported; it is not, because it would not be comparable with the
        ensemble spread of the learned model, and presenting two different
        notions of uncertainty under one name is worse than reporting none.
        """
        return np.zeros(_check_x(x).shape[0])


class LinkAvailabilityModel:
    """Learned predictor: bagged, calibrated histogram gradient boosting.

    Parameters
    ----------
    n_members : ensemble size (bootstrap resamples), >= 2.
    seed : master seed; member seeds derive from it deterministically.
    max_iter : boosting iterations per member, >= 1.
    max_depth : per-tree depth cap, >= 1.
    learning_rate : boosting learning rate, > 0.
    method : ``"sigmoid"`` (Platt) or ``"isotonic"``.
    calibration_folds : inner cross-validation folds for calibration, >= 2.
    """

    def __init__(self, n_members: int = 5, seed: int = 0, max_iter: int = 120,
                 max_depth: int = 3, learning_rate: float = 0.1,
                 method: str = "sigmoid", calibration_folds: int = 3) -> None:
        if n_members < 2:
            raise ValueError(f"n_members must be >= 2, got {n_members}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"method must be 'sigmoid' or 'isotonic', got {method!r}")
        if calibration_folds < 2:
            raise ValueError(
                f"calibration_folds must be >= 2, got {calibration_folds}")
        self.n_members = n_members
        self.seed = seed
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.method = method
        self.calibration_folds = calibration_folds
        self._members: list[CalibratedClassifierCV] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinkAvailabilityModel:
        """Train the bagged, calibrated ensemble."""
        x, y = _check_xy(x, y)
        if len(np.unique(y)) < 2:
            raise ValueError(
                "training labels are single-class; calibration is undefined. Enlarge the "
                "training split or change the split seed.")
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        self._members = []
        for _ in range(self.n_members):
            for _attempt in range(50):
                idx = rng.integers(0, n, size=n)
                if len(np.unique(y[idx])) == 2:
                    break
            else:  # pragma: no cover - needs a near-degenerate label vector
                raise RuntimeError(
                    "could not draw a two-class bootstrap resample in 50 attempts")
            base = HistGradientBoostingClassifier(
                max_iter=self.max_iter, max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=int(rng.integers(0, 2 ** 31 - 1)))
            member = CalibratedClassifierCV(base, method=self.method,
                                            cv=self.calibration_folds)
            member.fit(x[idx], y[idx])
            self._members.append(member)
        return self

    def _member_probs(self, x: np.ndarray) -> np.ndarray:
        if not self._members:
            raise RuntimeError("LinkAvailabilityModel.fit must be called first")
        x = _check_x(x)
        return np.stack([m.predict_proba(x)[:, 1] for m in self._members])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Ensemble-mean p(close) in [0, 1], shape ``(n,)``."""
        return self._member_probs(x).mean(axis=0)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        """Ensemble standard deviation of p(close): the epistemic spread.

        See the module docstring for what this does and does not represent.
        """
        return self._member_probs(x).std(axis=0, ddof=0)

    def predict_with_uncertainty(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: ``(mean, std)`` from one pass over the members."""
        probs = self._member_probs(x)
        return probs.mean(axis=0), probs.std(axis=0, ddof=0)


@dataclass(frozen=True)
class PredictorScores:
    """Verification summary for one predictor on one split."""

    name: str
    n: int
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    log_loss: float
    accuracy_at_half: float
    mean_forecast: float
    base_rate: float
    n_bins: int

    def format_row(self) -> str:
        """One fixed-width line for a comparison table."""
        return (f"{self.name:<22}{self.brier:>9.5f}{self.reliability:>9.5f}"
                f"{self.resolution:>9.5f}{self.ece:>9.5f}{self.log_loss:>9.5f}"
                f"{self.accuracy_at_half:>9.4f}")

    @staticmethod
    def header() -> str:
        """Matching header for :meth:`format_row`."""
        return (f"{'predictor':<22}{'Brier':>9}{'REL':>9}{'RES':>9}"
                f"{'ECE':>9}{'logloss':>9}{'acc@0.5':>9}")


def score_predictor(name: str, p: np.ndarray, y: np.ndarray, n_bins: int = 10,
                    ) -> PredictorScores:
    """Compute the full verification summary for one predictor."""
    dec = brier_decomposition(p, y, n_bins=n_bins)
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return PredictorScores(
        name=name, n=int(p.size), brier=dec.brier, reliability=dec.reliability,
        resolution=dec.resolution, uncertainty=dec.uncertainty,
        ece=expected_calibration_error(p, y, n_bins=n_bins),
        log_loss=log_loss_safe(p, y),
        accuracy_at_half=float(np.mean((p >= 0.5).astype(float) == y)),
        mean_forecast=float(p.mean()), base_rate=float(y.mean()), n_bins=n_bins)


def grouped_split(data: LinkDataset, test_fraction: float = 0.3, seed: int = 0,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices by LINK so no link appears on both sides.

    Returns ``(train_idx, test_idx)``.  Raises if either side would be empty
    or single-class, because both make the comparison meaningless.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    rng = np.random.default_rng(seed)
    groups = np.unique(data.group)
    perm = rng.permutation(groups.size)
    n_test = max(1, int(round(groups.size * test_fraction)))
    test_groups = set(groups[perm[:n_test]].tolist())
    mask = np.array([g in test_groups for g in data.group])
    test_idx = np.nonzero(mask)[0]
    train_idx = np.nonzero(~mask)[0]
    for label, idx in (("train", train_idx), ("test", test_idx)):
        if idx.size == 0:
            raise ValueError(f"grouped_split produced an empty {label} set")
        if len(np.unique(data.y[idx])) < 2:
            raise ValueError(
                f"grouped_split produced a single-class {label} set; change the seed "
                f"or the test_fraction")
    return train_idx, test_idx


def row_split(data: LinkDataset, test_fraction: float = 0.3, seed: int = 0,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise split, kept only to MEASURE the leakage a grouped split avoids.

    Not the split to report results on; see the module docstring.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    rng = np.random.default_rng(seed)
    n = len(data)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])
