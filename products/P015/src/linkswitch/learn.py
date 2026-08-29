"""Learned imminent-optical-outage predictor (the AI element of LinkSwitch).

Classical baselines (``FixedThresholdPolicy``, ``HysteresisPolicy`` in
``policies.py``) are implemented and validated first — see
``validation/VALIDATION.md`` and ``MODEL_CARD.md``. This module implements
the learned predictive policy's underlying classifier: scikit-learn only
(no PyTorch, per project constraints), trained on simulated telemetry.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import label_imminent_outage, rolling_features
from .scenario import Telemetry

__all__ = ["OutagePredictor", "build_training_set", "train_outage_predictor"]


class OutagePredictor:
    """Wraps a scikit-learn classification pipeline with a confidence output.

    ``predict_proba`` returns P(imminent outage) in [0, 1] for each row —
    the required "confidence output" for the learned policy (Level 2 /
    AI-product requirement). This is the model's own calibration-free
    class-1 probability estimate (a `RandomForestClassifier` vote fraction),
    not a formally calibrated probability; see MODEL_CARD.md.
    """

    def __init__(self, n_estimators: int = 40, max_depth: int = 4, random_state: int = 0):
        if (
            not isinstance(n_estimators, (int, np.integer))
            or isinstance(n_estimators, bool)
            or n_estimators < 1
        ):
            raise ValueError(f"n_estimators must be a positive integer, got {n_estimators!r}")
        if not isinstance(max_depth, (int, np.integer)) or isinstance(max_depth, bool) or max_depth < 1:
            raise ValueError(f"max_depth must be a positive integer, got {max_depth!r}")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self._pipeline: Pipeline | None = None
        self._classes_single: bool = False
        self._single_class_value: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OutagePredictor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y).astype(int)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x must be 2-D with the same number of rows as y; got x.shape={x.shape}, "
                f"y.shape={y.shape}"
            )
        if x.shape[0] < 2:
            raise ValueError("need at least 2 training rows")
        unique = np.unique(y)
        if unique.size < 2:
            # Degenerate but real scenario (e.g. a horizon/scenario combo with
            # no outages at all in the training data): fall back to a
            # constant predictor rather than raising, and say so plainly.
            self._classes_single = True
            self._single_class_value = int(unique[0])
            self._pipeline = None
            return self
        self._classes_single = False
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        self._pipeline.fit(x, y)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None or self._classes_single

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """P(imminent outage) for each row of ``x``, shape (n,)."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before predict_proba()")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if self._classes_single:
            return np.full(x.shape[0], float(self._single_class_value))
        proba = self._pipeline.predict_proba(x)
        classes = self._pipeline.named_steps["clf"].classes_
        col = int(np.where(classes == 1)[0][0])
        return proba[:, col]


def build_training_set(
    telemetries: list[Telemetry], tau_phys: float, horizon: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate causal features + imminent-outage labels across episodes.

    The last ``horizon`` steps of each episode are dropped: their labels are
    truncated by the end of the series (see ``label_imminent_outage``) and
    including them would bias training toward under-predicting outages.
    """
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if len(telemetries) == 0:
        raise ValueError("telemetries must be non-empty")
    xs, ys = [], []
    for tel in telemetries:
        feats = rolling_features(tel.irradiance, window)
        labels = label_imminent_outage(tel.irradiance, tau_phys, horizon)
        n = tel.n_steps
        keep = max(0, n - horizon)
        if keep == 0:
            continue
        xs.append(feats[:keep])
        ys.append(labels[:keep])
    if not xs:
        raise ValueError("no usable rows: every episode is shorter than the horizon")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_outage_predictor(
    telemetries: list[Telemetry],
    tau_phys: float,
    horizon: int,
    window: int,
    random_state: int = 0,
    n_estimators: int = 40,
    max_depth: int = 4,
) -> OutagePredictor:
    """Build the training set from simulated telemetry and fit the predictor."""
    tau_phys = float(tau_phys)
    if not (math.isfinite(tau_phys) and tau_phys > 0.0):
        raise ValueError(f"tau_phys must be finite and > 0, got {tau_phys!r}")
    x, y = build_training_set(telemetries, tau_phys, horizon, window)
    model = OutagePredictor(n_estimators=n_estimators, max_depth=max_depth,
                             random_state=random_state)
    model.fit(x, y)
    return model
