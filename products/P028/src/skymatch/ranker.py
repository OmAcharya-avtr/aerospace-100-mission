"""The learned candidate ranker: the AI element, and the thing being measured.

The classical decision rules in :mod:`skymatch.identify` are *hard rules*. The
Pyramid rule accepts a candidate when a fourth spot confirms it uniquely and
otherwise returns nothing, with no graded output and no operating point to
move. That is a design choice with a measurable consequence: it keeps the
false-identification rate very low and pays for it with the identification
rate in exactly the regimes where identification is hard.

This module replaces the *decision*, not the search. It scores each candidate
produced by the same geometric stage with a gradient-boosted classifier over
the 13 features of :data:`skymatch.identify.FEATURE_NAMES`, and accepts the
best-scoring one when its probability clears a threshold. The threshold is the
operating point the classical rule does not have, so the comparison is not one
number against another but a curve against a point --
``validation/validate_ml_vs_classical.py`` reports it that way.

Model: ``sklearn.ensemble.HistGradientBoostingClassifier``. PyTorch is not
available in the target environment and there are two CPU cores, so a deep
model was never an option; a gradient-boosted tree ensemble is also the right
shape for 13 heterogeneous hand-built features and tens of thousands of rows.
This is a constraint on the result, not evidence about what other
architectures would do.

Confidence: ``predict_proba``. It is a probability, so it can be checked
against outcomes rather than merely believed, and
``validation/validate_ml_vs_classical.py`` reports its reliability table,
Brier score and expected calibration error. An uncalibrated score would not be
usable as the confidence output mission section 11 requires; a measured one
might be, and the measurement says how far.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import HistGradientBoostingClassifier

from .identify import FEATURE_NAMES, MAGNITUDE_FEATURE_INDICES, Candidate

__all__ = ["LearnedRanker", "brier_score", "expected_calibration_error", "reliability_table"]

#: Batches smaller than this are scored on one thread. See :func:`_single_thread`.
SMALL_BATCH = 1000


@contextlib.contextmanager
def _single_thread(n_rows: int) -> Iterator[None]:
    """Score small batches on one thread.

    ``HistGradientBoostingClassifier.predict_proba`` parallelises over rows with
    OpenMP. A star-identification frame produces tens of candidate rows, and at
    that size the thread-pool entry and exit cost far more than the trees do.
    Measured on the 2-core target machine, one 25-row call takes **19.0 ms**
    with two OpenMP threads and **1.7 ms** with one -- an 11x penalty for using
    the second core, and a frame-level cost that swamped every other part of
    the matcher before this was pinned (``validation/VALIDATION.md`` section 7).

    Uses ``threadpoolctl``, which scikit-learn already depends on. If it is
    missing the limiter is a no-op and the only consequence is that small
    batches are slower.
    """
    if n_rows >= SMALL_BATCH:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        yield
        return
    with threadpool_limits(limits=1):
        yield


class LearnedRanker:
    """Score and rank candidate identifications.

    Parameters
    ----------
    use_magnitude_features
        If ``False``, columns :data:`skymatch.identify.MAGNITUDE_FEATURE_INDICES`
        are dropped. The simulator gives the instrument the catalogue's own
        magnitude scale plus Gaussian noise, which flatters photometric
        features; the ablation measures how much the ranker leans on them.
    max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization
        Passed to ``HistGradientBoostingClassifier``.
    random_state
        Seed for the classifier.
    """

    def __init__(
        self,
        use_magnitude_features: bool = True,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.use_magnitude_features = bool(use_magnitude_features)
        self.random_state = int(random_state)
        self._model = HistGradientBoostingClassifier(
            max_iter=int(max_iter),
            learning_rate=float(learning_rate),
            max_leaf_nodes=int(max_leaf_nodes),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.random_state,
        )
        self._fitted = False

    # -- feature handling -------------------------------------------------

    @property
    def columns(self) -> np.ndarray:
        """Indices of :data:`FEATURE_NAMES` this ranker uses."""
        keep = [
            i
            for i in range(len(FEATURE_NAMES))
            if self.use_magnitude_features or i not in MAGNITUDE_FEATURE_INDICES
        ]
        return np.array(keep, dtype=int)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of the columns this ranker uses."""
        return tuple(FEATURE_NAMES[i] for i in self.columns)

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._fitted

    def _select(self, features: ArrayLike) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have {len(FEATURE_NAMES)} columns "
                f"(skymatch.identify.FEATURE_NAMES), got {x.shape[1]}"
            )
        return x[:, self.columns]

    # -- training ---------------------------------------------------------

    def fit(self, features: ArrayLike, labels: ArrayLike) -> LearnedRanker:
        """Fit on ``(n, 13)`` features and ``(n,)`` 0/1 labels. Returns ``self``."""
        x = self._select(features)
        y = np.asarray(labels).reshape(-1).astype(int)
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"labels has length {y.shape[0]}, expected {x.shape[0]}")
        if x.shape[0] < 50:
            raise ValueError(f"need at least 50 rows to fit, got {x.shape[0]}")
        if set(np.unique(y).tolist()) - {0, 1}:
            raise ValueError("labels must be 0 or 1")
        if np.unique(y).size < 2:
            raise ValueError("labels must contain both classes")
        self._model.fit(x, y)
        self._fitted = True
        return self

    # -- inference --------------------------------------------------------

    def score(self, features: ArrayLike) -> np.ndarray:
        """Probability ``(n,)`` that each row is the correct identification."""
        if not self._fitted:
            raise RuntimeError("LearnedRanker.score called before fit")
        x = self._select(features)
        with _single_thread(x.shape[0]):
            return self._model.predict_proba(x)[:, 1]

    def score_candidates(self, candidates: list[Candidate]) -> np.ndarray:
        """Probabilities for a candidate list; empty array for an empty list."""
        if not candidates:
            return np.empty(0)
        return self.score(np.stack([c.features for c in candidates]))

    def decide(
        self, candidates: list[Candidate], threshold: float = 0.5
    ) -> tuple[Candidate | None, float]:
        """Accept the highest-scoring candidate if it clears ``threshold``.

        Returns ``(candidate, confidence)``, or ``(None, best_score)`` when
        nothing clears the threshold -- so a caller can see how close the
        frame came to being accepted.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        scores = self.score_candidates(candidates)
        if scores.size == 0:
            return None, 0.0
        best = int(np.argmax(scores))
        if scores[best] < threshold:
            return None, float(scores[best])
        return candidates[best], float(scores[best])

    def permutation_importance(
        self, features: ArrayLike, labels: ArrayLike, rng: np.random.Generator, n_repeats: int = 3
    ) -> np.ndarray:
        """Drop in average precision when each used column is shuffled.

        Simple permutation importance, computed on whatever data is passed --
        pass held-out data, or the number describes memorisation instead of
        signal.
        """
        from sklearn.metrics import average_precision_score

        x = np.asarray(features, dtype=float)
        y = np.asarray(labels).reshape(-1)
        base = average_precision_score(y, self.score(x))
        out = np.zeros(self.columns.size)
        for slot, col in enumerate(self.columns):
            drops = []
            for _ in range(n_repeats):
                shuffled = x.copy()
                shuffled[:, col] = rng.permutation(shuffled[:, col])
                drops.append(base - average_precision_score(y, self.score(shuffled)))
            out[slot] = float(np.mean(drops))
        return out


def brier_score(probabilities: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean squared error of a probabilistic prediction, in ``[0, 1]``; lower is better."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    o = np.asarray(outcomes, dtype=float).reshape(-1)
    if p.shape != o.shape:
        raise ValueError(f"shapes differ: {p.shape} and {o.shape}")
    if p.size == 0:
        raise ValueError("cannot score an empty array")
    return float(np.mean((p - o) ** 2))


def reliability_table(
    probabilities: ArrayLike, outcomes: ArrayLike, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predictions and compare mean predicted with observed frequency.

    Returns ``(mean_predicted, observed_frequency, count)``, one row per
    non-empty equal-width bin of ``[0, 1]``. A perfectly calibrated model has
    ``mean_predicted == observed_frequency`` in every bin.
    """
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    o = np.asarray(outcomes, dtype=float).reshape(-1)
    if p.shape != o.shape:
        raise ValueError(f"shapes differ: {p.shape} and {o.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.bincount(which, minlength=n_bins)
    used = counts > 0
    mean_p = np.bincount(which, weights=p, minlength=n_bins)[used] / counts[used]
    freq = np.bincount(which, weights=o, minlength=n_bins)[used] / counts[used]
    return mean_p, freq, counts[used]


def expected_calibration_error(
    probabilities: ArrayLike, outcomes: ArrayLike, n_bins: int = 10
) -> float:
    """Count-weighted mean ``|predicted - observed|`` over the reliability bins."""
    mean_p, freq, counts = reliability_table(probabilities, outcomes, n_bins)
    return float(np.sum(counts * np.abs(mean_p - freq)) / counts.sum())
