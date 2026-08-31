"""Learned B-dot gain scheduler.

What it learns
--------------
A regressor maps the observable feature vector of ``features.py`` to

    y = log10( k_best(scenario) / k_base )

where ``k_best`` is the best **constant** gain for that scenario found by an
offline grid search on the training scenarios (the "oracle" gain), and
``k_base`` is the single tuned constant gain of the ``FixedGainPolicy``
baseline.  Every control step of a training scenario is labelled with that
scenario's single oracle gain, so the learner is being asked: *given what the
magnetometer has seen so far, what constant gain would have been best for this
vehicle?*

That framing has a consequence worth stating plainly: the training target is
the best constant gain, so a perfect learner reproduces the constant-gain
oracle and nothing better.  At deployment the predicted gain is re-evaluated
periodically and therefore varies with time, which may do better or worse than
any constant gain.  The benchmark reports both, plus the oracle itself as an
upper bound on what this target can buy.

Confidence
----------
The model is a ``RandomForestRegressor``; the spread of the individual tree
predictions is used as an uncertainty estimate,

    sigma = std_over_trees( tree_i(x) )                       [dex]
    confidence = 1 / (1 + sigma / confidence_scale)           in (0, 1]

and the applied correction is shrunk toward zero by that confidence,

    k = k_base * 10 ** ( confidence * clip(y_hat, -L, +L) )

so an out-of-distribution input with disagreeing trees degrades gracefully to
the classical baseline instead of commanding an arbitrary gain.  ``L``
(``max_log_adjust``) is a hard safety clamp, not a tuning knob.

**This is an ensemble-spread heuristic, not a calibrated predictive
interval.**  No coverage calibration was performed; treat it as a relative
disagreement measure only.  See ``MODEL_CARD.md``.

Compute
-------
scikit-learn only, ``n_jobs=1``, a few hundred trees at most: the fit takes
well under a second on 2 cores.  No PyTorch, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestRegressor

from .features import N_FEATURES


@dataclass
class GainScheduler:
    """RandomForest gain scheduler with an ensemble-spread confidence output.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum tree depth; kept small because the training set is small.
    min_samples_leaf : int
        Minimum samples per leaf.
    random_state : int
        Seed; fixing it makes ``predict_gain`` bit-reproducible.
    max_log_adjust : float
        Hard clamp ``L`` on the applied log10 gain correction [dex].
    confidence_scale : float
        ``sigma`` at which confidence falls to 0.5 [dex].
    """

    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 4
    random_state: int = 0
    max_log_adjust: float = 1.0
    confidence_scale: float = 0.25
    model: RandomForestRegressor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.n_estimators) < 2:
            raise ValueError("n_estimators must be >= 2 for a spread estimate")
        if self.max_log_adjust <= 0.0:
            raise ValueError("max_log_adjust must be positive")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be positive")

    @property
    def fitted(self) -> bool:
        """True once ``fit`` has been called."""
        return self.model is not None

    def fit(self, x: ArrayLike, y: ArrayLike) -> GainScheduler:
        """Fit on feature rows ``x`` (n, 8) and log10 gain ratios ``y`` (n,).

        Raises
        ------
        ValueError
            On shape mismatch, wrong feature count, or non-finite entries.
        """
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float).ravel()
        if xa.ndim != 2 or xa.shape[1] != N_FEATURES:
            raise ValueError(
                f"x must have shape (n, {N_FEATURES}), got {xa.shape}"
            )
        if ya.shape[0] != xa.shape[0]:
            raise ValueError(
                f"x has {xa.shape[0]} rows but y has {ya.shape[0]} entries"
            )
        if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
            raise ValueError("x and y must be finite")
        if xa.shape[0] < 2:
            raise ValueError("need at least two training rows")
        self.model = RandomForestRegressor(
            n_estimators=int(self.n_estimators),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            random_state=int(self.random_state),
            n_jobs=1,
        )
        self.model.fit(xa, ya)
        return self

    def predict_with_uncertainty(
        self, x: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean log10 ratio, ensemble std)`` for feature rows ``x``.

        Both arrays have shape ``(n,)`` and units of dex (log10 gain ratio).
        """
        if self.model is None:
            raise ValueError("scheduler is not fitted; call fit() first")
        xa = np.atleast_2d(np.asarray(x, dtype=float))
        if xa.shape[1] != N_FEATURES:
            raise ValueError(
                f"x must have {N_FEATURES} columns, got {xa.shape[1]}"
            )
        if not np.all(np.isfinite(xa)):
            raise ValueError("x must be finite")
        preds = np.stack([t.predict(xa) for t in self.model.estimators_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def confidence(self, spread_dex: ArrayLike) -> NDArray[np.float64]:
        """Map ensemble spread [dex] to a confidence in ``(0, 1]``."""
        s = np.asarray(spread_dex, dtype=float)
        if np.any(s < 0.0):
            raise ValueError("spread must be non-negative")
        return 1.0 / (1.0 + s / self.confidence_scale)

    def predict_gain(self, x: ArrayLike, base_gain: float) -> tuple[float, float]:
        """Scheduled gain and its confidence for one feature vector.

        Parameters
        ----------
        x : array_like, shape (8,)
            Feature vector from ``features.TelemetryWindow.features``.
        base_gain : float
            ``k_base`` [A m^2 s T^-1], positive.

        Returns
        -------
        (gain, confidence)
            ``gain`` [A m^2 s T^-1], ``confidence`` in ``(0, 1]``.
        """
        if not np.isfinite(base_gain) or base_gain <= 0.0:
            raise ValueError(f"base_gain must be positive, got {base_gain}")
        mean, spread = self.predict_with_uncertainty(np.asarray(x).reshape(1, -1))
        conf = float(self.confidence(spread)[0])
        adj = float(np.clip(mean[0], -self.max_log_adjust, self.max_log_adjust))
        return float(base_gain * 10.0 ** (conf * adj)), conf

    def feature_importances(self) -> NDArray[np.float64]:
        """Impurity-based feature importances, one per feature.

        These are the scikit-learn defaults and are biased toward
        high-cardinality features; they indicate, they do not prove.
        """
        if self.model is None:
            raise ValueError("scheduler is not fitted; call fit() first")
        return np.asarray(self.model.feature_importances_, dtype=float)
