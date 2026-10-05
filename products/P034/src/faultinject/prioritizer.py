"""Learned campaign prioritiser with an uncertainty output.

Problem.  A campaign budget buys ``B`` executions out of a fault space of size
``N >> B``.  The prioritiser predicts the severity of untried cases from the
severities of the cases already executed, so the next execution can be spent
where severity is predicted to be high.

Model.  ``sklearn.ensemble.RandomForestRegressor``.  A forest was chosen for
three reasons that matter more than accuracy here: it needs no feature scaling
(the feature vector mixes one-hot indicators with normalised parameters), it
trains in milliseconds on the few dozen samples a warm-up provides, and the
spread of its member trees gives a usable *per-case* uncertainty without a
second model.  PyTorch is not available in the build container, so no neural
alternative was attempted.

Uncertainty.  :meth:`CampaignPrioritizer.predict` returns, per case, the mean
over trees, the standard deviation over trees, and an interval
``mean +- Z * std``.  This is the *disagreement* of the ensemble, which is a
proxy for epistemic uncertainty; it is **not** a calibrated predictive interval
for a new case, and the empirical coverage of the nominal 95 % interval is
measured and reported in MODEL_CARD.md rather than assumed.  Treat it as a
ranking aid for exploration, not a probability statement.

Features (25, layout fixed by :func:`feature_names`)
    0-15   one-hot over the sixteen fault kinds
    16-19  one-hot over the channels (pos, vel, u, bus)
    20     start step as a fraction of the run, in [0, 1]
    21     duration as a fraction of the run, clipped to [0, 1]
    22-24  up to three kind-specific parameters, each normalised to [0, 1] on
           its own declared scale; unused slots are 0.0

The one-hot kind block means most of the signal available to the model is "which
kind of fault is this", which is exactly the signal a human campaign engineer
also has.  The benchmark therefore also reports a non-learned *kind-mean*
ablation so the reader can see how much the model adds beyond that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .campaign import FaultCase
from .taxonomy import DURATION_FRAC, START_FRAC, kinds, spec

CHANNELS: tuple[str, ...] = ("pos", "vel", "u", "bus")
MAX_PARAMS = 3
Z95 = 1.959963984540054
"""Standard normal 97.5th percentile, for the nominal 95 % interval."""


def feature_names() -> tuple[str, ...]:
    """Names of the 25 features, in order."""
    names = [f"kind:{k.value}" for k in kinds()]
    names += [f"channel:{c}" for c in CHANNELS]
    names += ["start_frac", "duration_frac"]
    names += [f"param{i}" for i in range(MAX_PARAMS)]
    return tuple(names)


N_FEATURES = len(feature_names())


def encode(case: FaultCase) -> np.ndarray:
    """Feature vector of one case. Shape ``(N_FEATURES,)``, dtype float64."""
    x = np.zeros(N_FEATURES, dtype=np.float64)
    all_kinds = kinds()
    x[all_kinds.index(case.injection.kind)] = 1.0
    ch = case.injection.channel
    if ch not in CHANNELS:
        raise ValueError(f"unknown channel {ch!r}; expected one of {CHANNELS}")
    x[len(all_kinds) + CHANNELS.index(ch)] = 1.0
    base = len(all_kinds) + len(CHANNELS)
    x[base] = START_FRAC.normalise(case.injection.start_step / case.n_steps)
    x[base + 1] = DURATION_FRAC.normalise(case.injection.duration_steps / case.n_steps)
    sp = spec(case.injection.kind)
    params = case.injection.params
    if len(sp.params) > MAX_PARAMS:  # pragma: no cover - taxonomy has at most one
        raise ValueError(f"{case.injection.kind.value} has more than {MAX_PARAMS} parameters")
    for i, pspec in enumerate(sp.params):
        x[base + 2 + i] = pspec.normalise(params[pspec.name])
    return x


def encode_many(cases: Sequence[FaultCase]) -> np.ndarray:
    """Feature matrix, shape ``(len(cases), N_FEATURES)``."""
    if not cases:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.vstack([encode(c) for c in cases])


@dataclass(frozen=True)
class SeverityPrediction:
    """Per-case prediction with its ensemble uncertainty."""

    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __len__(self) -> int:
        return int(self.mean.shape[0])

    def acquisition(self, kappa: float = 0.0) -> np.ndarray:
        """``mean + kappa * std``. ``kappa = 0`` is pure exploitation."""
        return self.mean + kappa * self.std


class CampaignPrioritizer:
    """Random-forest severity predictor with an ensemble-spread uncertainty."""

    def __init__(
        self,
        n_estimators: int = 60,
        min_samples_leaf: int = 2,
        max_features: float | str = 0.6,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.model: RandomForestRegressor | None = None
        self.n_train = 0

    def fit(self, cases: Sequence[FaultCase], severities: Sequence[float]) -> CampaignPrioritizer:
        """Fit on executed cases and their observed severities."""
        if len(cases) != len(severities):
            raise ValueError(
                f"cases and severities must match: {len(cases)} vs {len(severities)}"
            )
        if len(cases) < 2:
            raise ValueError(f"need at least 2 training cases, got {len(cases)}")
        x = encode_many(cases)
        y = np.asarray(severities, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError("severities must all be finite")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(x, y)
        self.n_train = len(cases)
        return self

    def predict(self, cases: Sequence[FaultCase]) -> SeverityPrediction:
        """Mean, ensemble standard deviation and nominal 95 % interval."""
        if self.model is None:
            raise RuntimeError("prioritizer is not fitted; call fit() first")
        if not cases:
            empty = np.zeros(0)
            return SeverityPrediction(empty, empty, empty, empty)
        x = encode_many(cases)
        per_tree = np.stack([t.predict(x) for t in self.model.estimators_], axis=0)
        mean = per_tree.mean(axis=0)
        std = per_tree.std(axis=0, ddof=0)
        return SeverityPrediction(
            mean=mean,
            std=std,
            lower=np.clip(mean - Z95 * std, 0.0, 1.0),
            upper=np.clip(mean + Z95 * std, 0.0, 1.0),
        )

    def interval_coverage(
        self, cases: Sequence[FaultCase], severities: Sequence[float]
    ) -> tuple[float, float]:
        """``(empirical coverage of the nominal 95 % interval, mean interval width)``.

        A coverage well below 0.95 means the ensemble spread understates the
        error, which is the usual behaviour of tree-ensemble disagreement; the
        measured value belongs in the model card, not a claim of calibration.
        """
        pred = self.predict(cases)
        y = np.asarray(severities, dtype=np.float64)
        inside = (y >= pred.lower) & (y <= pred.upper)
        return float(inside.mean()), float((pred.upper - pred.lower).mean())

    def feature_importances(self) -> dict[str, float]:
        """Impurity-based importances by feature name, descending."""
        if self.model is None:
            raise RuntimeError("prioritizer is not fitted; call fit() first")
        pairs = zip(feature_names(), self.model.feature_importances_, strict=True)
        return dict(sorted(pairs, key=lambda kv: -kv[1]))
