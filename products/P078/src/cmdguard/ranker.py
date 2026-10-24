"""The AI component: a learned risk ranker, and the two orderings it must beat.

The problem the ranker addresses is not detection. Forward execution already
finds every violation against the declared model, exactly, in one linear pass
(:mod:`cmdguard.trace`), and on the sequences measured here it costs
milliseconds. What is expensive is the human review of a long sequence: a
reviewer given 400 commands and a dozen findings still has to decide which
commands to read. The ranker orders commands by the probability that a command
is *involved in* a violation -- either the step that trips or a step named in
the trace of one -- so that a reviewer with time for twenty commands reads the
twenty most likely to matter.

The two orderings it is scored against are the ones a reviewer already has, and
both are free:

:func:`irreversibility_ordering`
    Irreversible commands first, then by declared resource draw. One line of
    sorting over a flag somebody already wrote down.
:func:`resource_draw_ordering`
    By the largest declared draw as a fraction of its ceiling.

:func:`random_ordering` is the control: a seeded shuffle, which is what
precision at k looks like when the ordering carries no information.

**The flag ordering matching or beating the learned model is a result to
publish, not a reason to retune.** It is the specification's own prediction for
this product and it is reported as measured in
``validation/validate_ranker_output.txt`` and in the README, whichever way it
came out.

Uncertainty
-----------
:meth:`RiskRanker.score` returns a spread alongside the risk: the standard
deviation across the forest's per-tree probabilities for the positive class.
That is a dispersion measure over an ensemble and **not** a calibrated
confidence interval -- see Wager, Hastie & Efron 2014, "Confidence intervals
for random forests: the jackknife and the infinitesimal jackknife", *Journal of
Machine Learning Research* 15, for what a calibrated interval would require.
Its measured behaviour is reported in ``MODEL_CARD.md``.

Compute
-------
A random forest of 200 trees at depth 8 over a corpus of a few thousand command
occurrences and twenty features. Training is seconds on two cores, not minutes;
the measured figure is in ``validation/validate_ranker_output.txt``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES, feature_matrix
from .model import Case
from .trace import CheckReport

__all__ = [
    "RiskScores",
    "RiskRanker",
    "irreversibility_ordering",
    "resource_draw_ordering",
    "random_ordering",
    "precision_at_k",
    "recall_at_k",
    "involved_mask",
    "RankerMetrics",
]


@dataclass(frozen=True)
class RiskScores:
    """Per-step risk with an ensemble dispersion beside it."""

    case_name: str
    risk: np.ndarray
    spread: np.ndarray

    def __post_init__(self) -> None:
        if self.risk.shape != self.spread.shape:
            raise ValueError(
                f"risk shape {self.risk.shape} and spread shape {self.spread.shape} differ"
            )

    def order(self) -> np.ndarray:
        """Step indices by descending risk, ties broken by step order.

        Tie-breaking by step order is deliberate: it is what a reviewer with no
        further information does, and it makes precision at k reproducible.
        """
        return np.lexsort((np.arange(self.risk.size), -self.risk))

    def top_k(self, k: int) -> tuple[int, ...]:
        """The ``k`` highest-risk step indices, in ranked order."""
        if k < 1:
            raise ValueError(f"k={k} must be >= 1")
        return tuple(int(i) for i in self.order()[:k])


def involved_mask(report: CheckReport, n_steps: int) -> np.ndarray:
    """Boolean label vector: True for a step involved in any violation.

    "Involved" means the step either tripped a violation or appears in the
    provenance trace of one. That is the set a reviewer has to read in order to
    understand the findings, which is what the ranker is for.
    """
    mask = np.zeros(n_steps, dtype=bool)
    for index in report.involved_steps():
        if 0 <= index < n_steps:
            mask[index] = True
    return mask


def precision_at_k(order: Sequence[int], labels: np.ndarray, k: int) -> float:
    """Fraction of the top ``k`` ranked steps that are labelled involved.

    Returns ``nan`` when the sequence is shorter than ``k`` or carries no
    positive label, because precision at k over fewer than k items, or over a
    sequence with nothing to find, is not a number worth averaging. The caller
    reports how many cases were excluded for that reason; see
    :func:`cmdguard.benchmark.ranker_comparison`.
    """
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    if labels.size < k or not labels.any():
        return float("nan")
    chosen = list(order)[:k]
    hits = sum(1 for index in chosen if labels[index])
    return hits / float(k)


def recall_at_k(order: Sequence[int], labels: np.ndarray, k: int) -> float:
    """Fraction of all involved steps that appear in the top ``k``.

    Reported beside precision at k because it is the number a reviewer with a
    fixed budget actually cares about: of everything I would have had to read,
    how much did the first k entries cover. ``nan`` under the same conditions
    as :func:`precision_at_k`.
    """
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    if labels.size < k or not labels.any():
        return float("nan")
    chosen = list(order)[:k]
    hits = sum(1 for index in chosen if labels[index])
    return hits / float(labels.sum())


def irreversibility_ordering(case: Case) -> np.ndarray:
    """Irreversible commands first, then by declared draw fraction.

    The free baseline. One sort over a flag that is already in the command
    dictionary, which is why the specification predicts it will be hard to beat.
    """
    matrix = feature_matrix(case)
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    irreversible = matrix[:, FEATURE_NAMES.index("irreversible")]
    fraction = matrix[:, FEATURE_NAMES.index("draw_fraction_of_ceiling")]
    score = irreversible * 10.0 + fraction
    return np.lexsort((np.arange(score.size), -score))


def resource_draw_ordering(case: Case) -> np.ndarray:
    """By largest declared draw as a fraction of its declared ceiling."""
    matrix = feature_matrix(case)
    if matrix.size == 0:
        return np.zeros(0, dtype=int)
    fraction = matrix[:, FEATURE_NAMES.index("draw_fraction_of_ceiling")]
    return np.lexsort((np.arange(fraction.size), -fraction))


def random_ordering(case: Case, seed: int) -> np.ndarray:
    """Seeded shuffle. The control: precision at k with no information."""
    indices = list(range(len(case.sequence)))
    random.Random(seed).shuffle(indices)
    return np.asarray(indices, dtype=int)


@dataclass(frozen=True)
class RankerMetrics:
    """Training-side record kept for the model card."""

    n_cases: int
    n_rows: int
    n_positive: int
    positive_fraction: float
    n_estimators: int
    max_depth: int
    seed: int
    fit_seconds: float
    feature_importance: dict[str, float]
    features_used: tuple[str, ...] = FEATURE_NAMES
    features_dropped: tuple[str, ...] = ()


class RiskRanker:
    """Random-forest risk ranker over static command features.

    Parameters
    ----------
    n_estimators:
        Trees in the forest.
    max_depth:
        Depth cap. Eight on twenty features over a few thousand rows; deeper
        forests memorised the generator in a first pass and the held-out
        precision did not move.
    seed:
        Passed to ``random_state``, so a fit is reproducible.

    Notes
    -----
    ``class_weight="balanced"`` is set because the positive class is a minority
    -- the measured fraction is in :attr:`metrics` and in the model card -- and
    an unweighted forest on this corpus predicts the majority class for
    everything, which makes every ordering it produces a tie.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        seed: int = 20261024,
        drop_features: Sequence[str] = (),
    ) -> None:
        if n_estimators < 1:
            raise ValueError(f"n_estimators={n_estimators} must be >= 1")
        if max_depth < 1:
            raise ValueError(f"max_depth={max_depth} must be >= 1")
        unknown = sorted(set(drop_features) - set(FEATURE_NAMES))
        if unknown:
            raise ValueError(f"drop_features names no such feature: {', '.join(unknown)}")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.seed = int(seed)
        self.drop_features = tuple(sorted(drop_features))
        self._columns = tuple(
            index for index, name in enumerate(FEATURE_NAMES) if name not in set(drop_features)
        )
        if not self._columns:
            raise ValueError("drop_features would remove every feature")
        self._forest: RandomForestClassifier | None = None
        self.metrics: RankerMetrics | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names actually used, after ``drop_features``."""
        return tuple(FEATURE_NAMES[i] for i in self._columns)

    def _design(self, case: Case) -> np.ndarray:
        """Feature matrix restricted to the columns this ranker uses."""
        return feature_matrix(case)[:, list(self._columns)]

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._forest is not None

    def fit(self, pairs: Iterable[tuple[Case, CheckReport]]) -> RiskRanker:
        """Fit on ``(case, report)`` pairs. Labels come from the forward walk.

        The forward walk supplies the labels and nothing else: the features are
        computed from the declarations alone. See :mod:`cmdguard.features`.

        Raises
        ------
        ValueError
            When the training set has no rows, or only one class.
        """
        import time

        blocks: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        n_cases = 0
        for case, report in pairs:
            n_steps = len(case.sequence)
            if n_steps == 0:
                continue
            blocks.append(self._design(case))
            labels.append(involved_mask(report, n_steps))
            n_cases += 1
        if not blocks:
            raise ValueError("training set is empty: no case contributed a step")
        matrix = np.vstack(blocks)
        target = np.concatenate(labels)
        if target.all() or not target.any():
            raise ValueError(
                "training labels have a single class; a ranker cannot be fitted on a corpus "
                "in which every command is involved in a violation or none is"
            )
        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=1,
        )
        start = time.perf_counter()
        forest.fit(matrix, target)
        elapsed = time.perf_counter() - start
        self._forest = forest
        self.metrics = RankerMetrics(
            n_cases=n_cases,
            n_rows=int(matrix.shape[0]),
            n_positive=int(target.sum()),
            positive_fraction=float(target.mean()),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            seed=self.seed,
            fit_seconds=elapsed,
            features_used=self.feature_names,
            features_dropped=self.drop_features,
            feature_importance={
                name: float(value)
                for name, value in sorted(
                    zip(self.feature_names, forest.feature_importances_, strict=True),
                    key=lambda item: -item[1],
                )
            },
        )
        return self

    def score(self, case: Case) -> RiskScores:
        """Risk and ensemble dispersion per step of ``case``.

        Raises
        ------
        RuntimeError
            When called before :meth:`fit`.
        """
        if self._forest is None:
            raise RuntimeError("RiskRanker.score called before fit")
        n_steps = len(case.sequence)
        if n_steps == 0:
            return RiskScores(case.name, np.zeros(0), np.zeros(0))
        matrix = self._design(case)
        positive_index = int(np.where(self._forest.classes_)[0][0])
        per_tree = np.vstack(
            [tree.predict_proba(matrix)[:, positive_index] for tree in self._forest.estimators_]
        )
        return RiskScores(
            case_name=case.name,
            risk=per_tree.mean(axis=0),
            spread=per_tree.std(axis=0, ddof=0),
        )

    def order(self, case: Case) -> np.ndarray:
        """Step indices by descending learned risk."""
        return self.score(case).order()
