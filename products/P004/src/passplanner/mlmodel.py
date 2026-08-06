"""AI availability model: pass-success probability with uncertainty.

Baseline (implemented FIRST, per mission rules): the climatological prior --
predict the monthly clear-sky probability, ignoring weather features.  The ML
model must beat this baseline on held-out Brier score to be worth shipping;
measured numbers are in validation/VALIDATION.md and MODEL_CARD.md.

Model: a bagged ensemble of ``sklearn.ensemble.GradientBoostingClassifier``
members (bootstrap resamples, distinct seeds).  The ensemble mean is the
pass-success probability; the ensemble standard deviation is the confidence
output required for AI products (epistemic spread only -- it does NOT capture
irreducible weather randomness, which the probability itself expresses).

Trained exclusively on the synthetic dataset from :mod:`passplanner.synthdata`
(see DATASET_CARD.md).  This model is not certified for operational flight
use.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .synthdata import FEATURE_NAMES


class ClimatologyBaselineModel:
    """Baseline: p(success) = climatological monthly prior (feature 0).

    Stateless; ``fit`` exists for interface parity and validates shapes.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClimatologyBaselineModel":
        x, _ = _check_xy(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return p(success) in [0, 1], shape (n,)."""
        x = _check_x(x)
        return x[:, 0].astype(float)


class PassSuccessModel:
    """Bagged gradient-boosting classifier with an uncertainty output.

    Parameters
    ----------
    n_members : ensemble size (bootstrap resamples), >= 2.
    seed : master seed; member seeds are derived deterministically.
    n_estimators, max_depth, learning_rate : per-member GBM hyperparameters
        (defaults sized for the < 3 min / 2-core compute budget).
    """

    def __init__(self, n_members: int = 5, seed: int = 0,
                 n_estimators: int = 150, max_depth: int = 2,
                 learning_rate: float = 0.1):
        if n_members < 2:
            raise ValueError(f"n_members must be >= 2, got {n_members}")
        self.n_members = n_members
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._members: list[GradientBoostingClassifier] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PassSuccessModel":
        """Train the ensemble on features x (n, 7) and binary labels y (n,)."""
        x, y = _check_xy(x, y)
        rng = np.random.default_rng(self.seed)
        self._members = []
        n = x.shape[0]
        for k in range(self.n_members):
            idx = rng.integers(0, n, size=n)  # bootstrap resample
            member = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=int(rng.integers(0, 2**31 - 1)))
            member.fit(x[idx], y[idx])
            self._members.append(member)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Ensemble-mean pass-success probability, shape (n,), in [0, 1]."""
        p, _sigma = self.predict_with_uncertainty(x)
        return p

    def predict_with_uncertainty(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_mean, p_std): probability and ensemble spread, each (n,).

        p_std is the standard deviation of member probabilities -- a
        confidence/uncertainty output (epistemic spread across bootstrap
        members).  Large p_std flags inputs where the model is unreliable
        (e.g. feature combinations rare in training data).
        """
        if not self._members:
            raise RuntimeError("model is not fitted; call fit() first")
        x = _check_x(x)
        probs = np.stack([m.predict_proba(x)[:, 1] for m in self._members])
        return probs.mean(axis=0), probs.std(axis=0)


def _check_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"x must have shape (n, {len(FEATURE_NAMES)}) with columns {FEATURE_NAMES}, "
            f"got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    return x


def _check_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _check_x(x)
    y = np.asarray(y)
    if y.shape != (x.shape[0],):
        raise ValueError(f"y must have shape ({x.shape[0]},), got {y.shape}")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y must contain only binary labels 0/1")
    return x, y.astype(int)
