"""The learned detector: a windowed-feature classifier scored at equal ARL0.

This is the AI component, and it is built and measured **after** the five
analytic detectors, against them, at a matched measured ARL0. The comparison at
equal operating point is the only one that means anything; a learned detector
that fires more often will of course detect sooner.

Architecture
------------
``sklearn.ensemble.RandomForestClassifier`` on the eight windowed features in
:mod:`telemdrift.features`. Score is ``predict_proba(...)[:, 1]``, the forest's
vote fraction for "a change has occurred in or just before this window". The
calibrated scalar is the probability threshold ``p*``; an alarm is raised when
the score exceeds it, after which the detector enters a refractory period of
``window`` samples so one change cannot produce ``window`` alarms.

Why a random forest and not something deeper
--------------------------------------------
PyTorch is not available in the build container (ADR-005 amendment) and the
container has two cores. Eight features and a few thousand windows is a problem
a forest fits in under a second; there is no argument that a deeper model is
needed and no evidence here that would support one.

Confidence output
-----------------
``score`` is the forest's vote fraction, which is the model's confidence output
and is exposed by :meth:`LearnedDetector.last_score`. It is *not* claimed to be
calibrated. ``validation/validate_learned.py`` measures its reliability against
observed frequency and reports the gap. A sibling product in this portfolio
(P056 CalibAudit) is the one that recalibrates such a score; this product only
measures that it needs it.

Training-label convention, which is a modelling choice and not a fact
---------------------------------------------------------------------
A window is labelled 1 when its right edge lies in ``[change_index,
change_index + horizon)`` and 0 otherwise. ``horizon`` therefore *defines* what
the model is being asked to detect, and the ARL1 it achieves cannot be better
than the label convention allows. ``horizon = window`` is used so that a
positive window always contains at least one post-change sample.

Negative classes used for training are declared in
:func:`build_training_set` and matter more than the architecture: the model
never sees a transient spike during training unless asked to, so the transient
experiment is held out for every detector on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .detectors import Detector
from .features import WINDOW, window_features
from .streams import ChangeSpec, change_stream, stationary

__all__ = [
    "LearnedDetector",
    "TrainingSet",
    "build_training_set",
    "score_stream",
    "train_learned_detector",
]


@dataclass(frozen=True)
class TrainingSet:
    """Feature matrix, labels and the provenance needed to regenerate them."""

    features: np.ndarray
    labels: np.ndarray
    seeds: tuple[int, ...]
    window: int
    horizon: int
    change_specs: tuple[tuple[str, float], ...]
    include_transients: bool

    def summary(self) -> str:
        pos = int(self.labels.sum())
        return (
            f"{self.features.shape[0]} windows x {self.features.shape[1]} features, "
            f"{pos} positive ({100 * pos / max(self.labels.size, 1):.1f} %), "
            f"window={self.window}, horizon={self.horizon}, "
            f"transients_in_training={self.include_transients}, "
            f"{len(self.seeds)} seeds"
        )


#: Change specifications used for the positive class. Magnitudes span the range
#: the benchmark scores, so the model is not trained on one severity and tested
#: on another.
TRAIN_CHANGES: tuple[tuple[str, float], ...] = (
    ("mean_step", 0.5),
    ("mean_step", 1.0),
    ("mean_step", 2.0),
    ("variance_step", 1.8),
    ("variance_step", 3.0),
    ("drift_ramp", 0.01),
    ("drift_ramp", 0.03),
)


def build_training_set(
    seeds,
    window: int = WINDOW,
    horizon: int | None = None,
    pre_length: int = 400,
    post_length: int = 400,
    n_stationary: int = 6,
    stationary_length: int = 800,
    include_transients: bool = False,
    transient_amplitude: float = 4.0,
    transient_duration: int = 20,
) -> TrainingSet:
    """Generate a labelled window set from seeded streams.

    Parameters
    ----------
    seeds:
        One seed per (change type, magnitude) replicate. The same seed list is
        reused across change specs with a per-spec offset so the streams differ.
    include_transients:
        When ``True``, transient-spike streams are added as negatives. This is
        **off** for the headline comparison, so that the transient experiment is
        held out for every detector equally; ``validate_transient.py`` runs it
        both ways and reports what the extra supervision buys.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    h = window if horizon is None else int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    feats: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    for spec_i, (kind, mag) in enumerate(TRAIN_CHANGES):
        for s in seeds:
            seed = int(s) + 10_000 * (spec_i + 1)
            stream, idx = change_stream(
                pre_length, post_length, ChangeSpec(kind, mag), seed
            )
            f = window_features(stream, window)
            right_edge = np.arange(f.shape[0]) + window - 1
            y = ((right_edge >= idx) & (right_edge < idx + h)).astype(int)
            feats.append(f)
            labs.append(y)
    for j in range(n_stationary):
        for s in seeds:
            seed = int(s) + 900_000 + 1000 * j
            f = window_features(stationary(stationary_length, seed), window)
            feats.append(f)
            labs.append(np.zeros(f.shape[0], dtype=int))
    if include_transients:
        for j in range(n_stationary):
            for s in seeds:
                seed = int(s) + 700_000 + 1000 * j
                stream, _ = change_stream(
                    pre_length,
                    post_length,
                    ChangeSpec("transient", transient_amplitude, transient_duration),
                    seed,
                )
                f = window_features(stream, window)
                feats.append(f)
                labs.append(np.zeros(f.shape[0], dtype=int))
    X = np.vstack(feats)
    y = np.concatenate(labs)
    return TrainingSet(
        features=X,
        labels=y,
        seeds=tuple(int(s) for s in seeds),
        window=window,
        horizon=h,
        change_specs=TRAIN_CHANGES,
        include_transients=include_transients,
    )


def train_learned_detector(
    training: TrainingSet,
    n_estimators: int = 120,
    max_depth: int = 8,
    random_state: int = 58_000,
) -> RandomForestClassifier:
    """Fit the forest. ``n_jobs`` is left at 1 deliberately.

    ``n_jobs > 1`` is 5.7-8.9x **slower** than ``n_jobs = 1`` at single-row
    inference on this container (a defect recorded by an earlier session in this
    portfolio and re-measured for this product in
    ``validation/outputs/validate_learned.txt``). Since the per-sample online
    cost of this detector is one of the numbers this product publishes, the
    model is fitted and benchmarked at ``n_jobs = 1`` throughout, and the fit is
    fast enough that nothing is lost.
    """
    if training.features.shape[0] != training.labels.size:
        raise ValueError("feature/label length mismatch")
    if training.labels.sum() == 0:
        raise ValueError("training set contains no positive windows")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(training.features, training.labels)
    model.n_jobs = 1
    return model


def score_stream(
    model: RandomForestClassifier, stream: np.ndarray, window: int = WINDOW
) -> np.ndarray:
    """Causal score series aligned to the stream.

    Returns
    -------
    np.ndarray
        Same length as ``stream``. Entry ``t`` is the model score from the
        window ending at ``t``, or 0.0 for ``t < window - 1`` where no full
        window exists. Because every feature is a function of the trailing
        window only, this batch computation is identical to the online one; the
        equality is pinned by ``tests/test_learned.py``.
    """
    x = np.asarray(stream, dtype=float)
    out = np.zeros(x.size)
    f = window_features(x, window)
    if f.shape[0] == 0:
        return out
    out[window - 1 :] = model.predict_proba(f)[:, 1]
    return out


class LearnedDetector(Detector):
    """Streaming wrapper around the fitted forest, with a refractory period.

    Interface-compatible with the analytic detectors so the same ARL0/ARL1
    machinery scores it. ``update`` is the honest online path and is what the
    per-sample cost figure measures; :func:`score_stream` plus
    :meth:`alarms_from_scores` is the identical-result batch path the Monte
    Carlo uses, because 2 cores cannot afford 10^6 single-row forest calls.
    """

    name = "Learned RF"
    threshold_name = "p*"

    def __init__(
        self,
        model: RandomForestClassifier,
        p_threshold: float = 0.5,
        window: int = WINDOW,
    ):
        if not 0.0 < p_threshold < 1.0:
            raise ValueError("p_threshold must satisfy 0 < p* < 1")
        if window < 4:
            raise ValueError("window must be >= 4")
        self.model = model
        self._p = float(p_threshold)
        self.window = int(window)
        self._buf = np.zeros(self.window)
        self._filled = 0
        self._pos = 0
        # Warm-up is expressed as a refractory period of ``window - 1`` so that
        # the first scorable sample is at stream index ``window - 1``, exactly
        # where :func:`score_stream` places its first score. The two paths must
        # agree index for index; ``tests/test_learned.py`` pins the identity.
        self._refractory = self.window - 1
        self._last_score = 0.0

    @property
    def threshold(self) -> float:
        return self._p

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("p_threshold must satisfy 0 < p* < 1")
        self._p = float(value)

    @staticmethod
    def default_threshold() -> float:
        """``p* = 0.5``: the argmax decision rule, which is the usual default."""
        return 0.5

    @property
    def last_score(self) -> float:
        """The model's most recent vote fraction: its confidence output."""
        return self._last_score

    def is_armed(self) -> bool:
        return self._refractory == 0

    def alarm_ratio(self) -> float:
        return self._last_score / self._p

    def reset(self) -> None:
        self._filled = 0
        self._pos = 0
        self._refractory = self.window - 1
        self._last_score = 0.0

    def update(self, x: float) -> bool:
        self._buf[self._pos] = float(x)
        self._pos = (self._pos + 1) % self.window
        if self._filled < self.window:
            self._filled += 1
        if self._refractory > 0:
            self._refractory -= 1
            return False
        ordered = np.concatenate([self._buf[self._pos :], self._buf[: self._pos]])
        from .features import window_features_single

        self._last_score = float(
            self.model.predict_proba(window_features_single(ordered))[0, 1]
        )
        if self._last_score > self._p:
            # An alarm is a reset: the next score is computed from a window of
            # ``window`` entirely post-alarm samples, which is what
            # :meth:`reset` also produces. The two must agree, because the ARL0
            # harness resets on alarm and the ARL1 path does not.
            self._refractory = self.window - 1
            return True
        return False

    def alarms_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Alarm indices from a precomputed score series, same logic as ``update``.

        Parameters
        ----------
        scores:
            Output of :func:`score_stream`, aligned to the stream.

        Returns
        -------
        np.ndarray
            Integer indices at which an alarm is raised.
        """
        s = np.asarray(scores, dtype=float)
        out: list[int] = []
        refractory = self.window - 1
        for t in range(s.size):
            if refractory > 0:
                refractory -= 1
                continue
            if s[t] > self._p:
                out.append(t)
                refractory = self.window - 1
        return np.asarray(out, dtype=int)
