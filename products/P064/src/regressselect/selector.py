"""The learned selector: a ranked order with a probability per test.

Architecture: gradient-boosted decision trees
(:class:`sklearn.ensemble.GradientBoostingClassifier`) over the eleven change
features in :mod:`regressselect.features`, trained on (revision, test node)
pairs from the *earlier* part of a commit history and evaluated on the later
part. The positive class is "this test failed at this revision".

Why trees and not a linear model: three of the features interact
multiplicatively with the distance feature (a large churn in a module two hops
away is not the same signal as the same churn four hops away), and a shallow
tree ensemble represents that without the interaction terms having to be
written out. PyTorch is not available in this environment and is not needed at
this data scale.

The confidence output is the class-one probability from
:meth:`sklearn.ensemble.GradientBoostingClassifier.predict_proba`. It is a
score, not a calibrated probability unless ``calibrate=True`` is passed, which
wraps the estimator in :class:`sklearn.calibration.CalibratedClassifierCV`
with an isotonic link fitted on a held-out slice of the *training* history.
Calibration quality is measured, not asserted: see the Brier score and the
reliability table in ``MODEL_CARD.md``.

Compute: the default configuration trains on 5 040 rows in well under one
second on two cores. Nothing is pickled or committed, so there is no
serialised-artifact compatibility surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from .features import FEATURE_NAMES, build_design_matrix, revision_features
from .history import History, Revision
from .importgraph import ImportGraph
from .prioritise import CoFailurePrioritiser, RankedTest

__all__ = ["LearnedSelector", "SelectorConfig"]


@dataclass(frozen=True)
class SelectorConfig:
    """Hyper-parameters of the learned selector. All dimensionless."""

    n_estimators: int = 150
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 1.0
    random_state: int = 20261015
    calibrate: bool = False
    calibration_folds: int = 3

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must lie in (0, 1]")
        if self.calibration_folds < 2:
            raise ValueError("calibration_folds must be at least 2")


@dataclass
class LearnedSelector:
    """Learned ranked selector over change features.

    Typical use::

        train, test = history.split_chronological(0.70)
        sel = LearnedSelector().fit(train, graph)
        ranked = sel.rank(test.revisions[0])
    """

    config: SelectorConfig = field(default_factory=SelectorConfig)
    graph: ImportGraph | None = field(default=None, repr=False)
    prioritiser: CoFailurePrioritiser | None = field(default=None, repr=False)
    model: object | None = field(default=None, repr=False)
    node_ids: tuple[str, ...] = field(default_factory=tuple, repr=False)
    n_train_rows: int = 0
    n_train_positives: int = 0

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names in column order."""
        return FEATURE_NAMES

    def fit(self, history: History, graph: ImportGraph) -> LearnedSelector:
        """Train on ``history``. Returns ``self``.

        ``history`` must be the *training* slice only. The internal
        co-failure estimator that supplies two of the features is fitted on
        the same slice, so no test-slice outcome reaches the features.
        """
        if len(history) == 0:
            raise ValueError("cannot fit on an empty history")
        self.graph = graph
        self.prioritiser = CoFailurePrioritiser().fit(history)
        self.node_ids = tuple(history.test_ids)
        x, y, _ = build_design_matrix(history, graph, self.prioritiser)
        if y.sum() == 0:
            raise ValueError(
                "the training history contains no failing test; a classifier "
                "cannot be fitted and the honest answer is to use a baseline"
            )
        base = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )
        if self.config.calibrate:
            self.model = CalibratedClassifierCV(
                base, method="isotonic", cv=self.config.calibration_folds
            ).fit(x, y)
        else:
            self.model = base.fit(x, y)
        self.n_train_rows = int(x.shape[0])
        self.n_train_positives = int(y.sum())
        return self

    def _require_fitted(self) -> None:
        if self.model is None or self.graph is None or self.prioritiser is None:
            raise ValueError("selector is not fitted; call fit(history, graph) first")

    def probabilities(self, revision: Revision, nodes: Sequence[str] | None = None) -> np.ndarray:
        """Class-one probability per test node, in the order of ``nodes``."""
        self._require_fitted()
        universe = tuple(nodes) if nodes is not None else self.node_ids
        x = revision_features(revision, universe, self.graph, self.prioritiser)
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)

    def rank(self, revision: Revision, nodes: Sequence[str] | None = None) -> list[RankedTest]:
        """Total order over the suite, highest probability first.

        Ties break on the node id ascending, so the order is deterministic.
        """
        universe = list(nodes) if nodes is not None else list(self.node_ids)
        probs = self.probabilities(revision, universe)
        pairs = sorted(zip(probs, universe, strict=True), key=lambda p: (-p[0], p[1]))
        return [RankedTest(node_id=n, score=float(p)) for p, n in pairs]

    def feature_importances(self) -> dict[str, float]:
        """Gradient-boosting feature importances, dimensionless, summing to 1.

        Raises when the selector was fitted with ``calibrate=True``, because
        the calibrated wrapper holds an ensemble of estimators and a single
        importance vector would misrepresent it.
        """
        self._require_fitted()
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError(
                "feature importances are unavailable for a calibrated selector; "
                "fit with calibrate=False to inspect them"
            )
        values = np.asarray(self.model.feature_importances_, dtype=float)
        return dict(zip(FEATURE_NAMES, (float(v) for v in values), strict=True))

    def brier_score(self, history: History) -> float:
        """Mean squared error of the probability against the outcome.

        Dimensionless, in [0, 1], lower is better. Computed over every
        (revision, test node) pair in ``history``.
        """
        self._require_fitted()
        x, y, _ = build_design_matrix(history, self.graph, self.prioritiser)
        probs = np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)
        return float(np.mean((probs - y) ** 2))
