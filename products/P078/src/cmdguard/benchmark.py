"""The two measurements this product exists to publish.

:func:`lint_miss_rate`
    How much of the forward walk's finding set a per-command static lint
    misses, over a seeded family of injected defects, broken down by defect
    family and by violation kind. The specification calls this a deliverable
    and it is one: a mission that relies on a per-command lint is entitled to a
    number rather than a warning.

:func:`ranker_comparison`
    Precision at k for the learned risk ranker against the irreversibility-flag
    ordering, the resource-draw ordering and a seeded random control, on
    held-out cases generated from seeds the ranker never saw. Reported as
    measured, in whichever direction it came out.

Both functions return plain dataclasses with every count they used, so that the
validation scripts print counts rather than only ratios. A ratio with no
numerator and denominator beside it cannot be checked by a reader.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .dataset import CorpusEntry
from .lint import LintStrength, static_lint
from .model import Case
from .ranker import (
    RiskRanker,
    involved_mask,
    irreversibility_ordering,
    precision_at_k,
    random_ordering,
    recall_at_k,
    resource_draw_ordering,
)
from .trace import ORDERING_KINDS, CheckReport

__all__ = [
    "FamilyMissRate",
    "MissRateResult",
    "lint_miss_rate",
    "OrderingScore",
    "RankerComparison",
    "ranker_comparison",
    "timing_table",
]


@dataclass
class FamilyMissRate:
    """Per-family counts. Every ratio in the report is derived from these."""

    family: str
    n_cases: int = 0
    n_noop_injections: int = 0
    n_cases_without_violation: int = 0
    n_truth: int = 0
    n_missed: int = 0
    n_truth_ordering: int = 0
    n_missed_ordering: int = 0
    n_lint_findings: int = 0
    n_lint_false_positive: int = 0
    missed_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def miss_rate(self) -> float:
        """Missed violations over all violations. ``nan`` when nothing to miss."""
        return self.n_missed / self.n_truth if self.n_truth else float("nan")

    @property
    def ordering_miss_rate(self) -> float:
        """Missed ordering-only violations over all of them."""
        return (
            self.n_missed_ordering / self.n_truth_ordering
            if self.n_truth_ordering
            else float("nan")
        )

    @property
    def false_positive_rate(self) -> float:
        """Lint findings the forward walk does not confirm, over all findings."""
        return (
            self.n_lint_false_positive / self.n_lint_findings
            if self.n_lint_findings
            else float("nan")
        )


@dataclass
class MissRateResult:
    """Aggregate and per-family miss rates for one lint strength."""

    strength: LintStrength
    families: dict[str, FamilyMissRate]
    total: FamilyMissRate
    seconds: float

    def rows(self) -> list[tuple[str, int, int, int, float, float, float]]:
        """``(family, truth, missed, false positives, miss, ordering miss, fp)``."""
        out = []
        for family in sorted(self.families):
            record = self.families[family]
            out.append(
                (
                    family,
                    record.n_truth,
                    record.n_missed,
                    record.n_lint_false_positive,
                    record.miss_rate,
                    record.ordering_miss_rate,
                    record.false_positive_rate,
                )
            )
        return out


def lint_miss_rate(
    corpus: Iterable[CorpusEntry],
    strength: LintStrength = LintStrength.PER_COMMAND,
) -> MissRateResult:
    """Measure what the static lint misses against forward execution.

    A violation and a lint finding are the same finding when they agree on
    ``(step, kind)``. That identity is deliberately strict: a lint that reports
    a precondition violation at the right step for the wrong reason is counted
    as having found it, which favours the baseline.

    Parameters
    ----------
    corpus:
        Entries from :func:`cmdguard.dataset.build_corpus`. Entries whose
        injection was a no-op, and entries whose forward walk found nothing,
        are counted and excluded from the ratio rather than silently dropped.
    strength:
        Which lint to measure. See :class:`cmdguard.lint.LintStrength`.

    Returns
    -------
    MissRateResult
        Per-family and aggregate counts.
    """
    start = time.perf_counter()
    families: dict[str, FamilyMissRate] = {}
    total = FamilyMissRate(family="all")

    for entry in corpus:
        family = entry.record.family if entry.record is not None else "clean"
        record = families.setdefault(family, FamilyMissRate(family=family))
        record.n_cases += 1
        total.n_cases += 1
        if entry.record is not None and not entry.record.applied:
            record.n_noop_injections += 1
            total.n_noop_injections += 1

        truth = entry.report
        lint = static_lint(entry.case, strength)
        truth_ids = {(v.step, v.kind.value): v for v in truth.violations}
        lint_ids = lint.identities()
        if not truth_ids:
            record.n_cases_without_violation += 1
            total.n_cases_without_violation += 1

        missed = [key for key in truth_ids if key not in lint_ids]
        ordering_keys = {k.value for k in ORDERING_KINDS}
        truth_ordering = [k for k in truth_ids if k[1] in ordering_keys]
        missed_ordering = [k for k in missed if k[1] in ordering_keys]
        false_positive = [key for key in lint_ids if key not in truth_ids]

        for target in (record, total):
            target.n_truth += len(truth_ids)
            target.n_missed += len(missed)
            target.n_truth_ordering += len(truth_ordering)
            target.n_missed_ordering += len(missed_ordering)
            target.n_lint_findings += len(lint_ids)
            target.n_lint_false_positive += len(false_positive)
            for _, kind in missed:
                target.missed_by_kind[kind] = target.missed_by_kind.get(kind, 0) + 1

    for target in (*families.values(), total):
        target.missed_by_kind = dict(sorted(target.missed_by_kind.items()))

    return MissRateResult(
        strength=strength,
        families=families,
        total=total,
        seconds=time.perf_counter() - start,
    )


@dataclass(frozen=True)
class OrderingScore:
    """Mean precision and recall at k for one ordering, with its counts."""

    label: str
    k: int
    mean_precision: float
    mean_recall: float
    n_scored: int
    n_excluded: int
    per_case_precision: tuple[float, ...] = ()

    def render(self) -> str:
        """One table row."""
        return (
            f"{self.label:<22s} k={self.k:<3d} precision={self.mean_precision:.6f} "
            f"recall={self.mean_recall:.6f} scored={self.n_scored} excluded={self.n_excluded}"
        )


@dataclass(frozen=True)
class RankerComparison:
    """Precision at k for the learned ranker and the orderings it must beat."""

    ks: tuple[int, ...]
    scores: dict[tuple[str, int], OrderingScore]
    n_train_cases: int
    n_test_cases: int
    positive_fraction_test: float
    fit_seconds: float
    score_seconds: float
    features_dropped: tuple[str, ...]
    feature_importance: dict[str, float]
    mean_involved_per_case: float = float("nan")
    mean_steps_per_case: float = float("nan")
    bootstrap_seed: int = 20261024

    def paired_bootstrap(
        self, label_a: str, label_b: str, k: int, n_boot: int = 4000
    ) -> tuple[float, float, float]:
        """Paired mean difference ``a - b`` at ``k`` with a 95 per cent interval.

        The pairing is by case: both orderings are scored on the same held-out
        sequences, so the difference is taken case by case and the interval is
        a percentile bootstrap over cases. This is what decides whether a
        margin of a few hundredths is a result or a coincidence, and the
        README quotes the interval rather than the margin alone.

        Returns
        -------
        tuple
            ``(mean difference, 2.5th percentile, 97.5th percentile)``.
        """
        left = self.scores.get((label_a, k))
        right = self.scores.get((label_b, k))
        if left is None or right is None:
            raise KeyError(f"no score for {label_a!r} or {label_b!r} at k={k}")
        a = np.asarray(left.per_case_precision, dtype=np.float64)
        b = np.asarray(right.per_case_precision, dtype=np.float64)
        if a.size != b.size or a.size == 0:
            raise ValueError(
                f"paired bootstrap needs equal, non-empty per-case vectors, got "
                f"{a.size} and {b.size}"
            )
        difference = a - b
        rng = np.random.default_rng(self.bootstrap_seed)
        draws = rng.integers(0, difference.size, size=(n_boot, difference.size))
        means = difference[draws].mean(axis=1)
        return (
            float(difference.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        )

    def best_at(self, k: int) -> str:
        """Label of the ordering with the highest mean precision at ``k``.

        Ties are reported as a tie rather than resolved: two orderings whose
        mean precision agrees to within 1e-12 on this corpus are not
        distinguished by this corpus.
        """
        candidates = [
            (score.mean_precision, label)
            for (label, kk), score in self.scores.items()
            if kk == k and score.mean_precision == score.mean_precision
        ]
        if not candidates:
            return "none"
        best = max(candidates)[0]
        winners = sorted(label for value, label in candidates if abs(value - best) < 1e-12)
        return winners[0] if len(winners) == 1 else "tie: " + ", ".join(winners)

    def table(self) -> list[str]:
        """Rendered table rows, grouped by k."""
        lines: list[str] = []
        for k in self.ks:
            for label in ("learned", "irreversibility_flag", "resource_draw", "random_control"):
                score = self.scores.get((label, k))
                if score is not None:
                    lines.append(score.render())
            for baseline in ("irreversibility_flag", "resource_draw"):
                mean, low, high = self.paired_bootstrap("learned", baseline, k)
                verdict = (
                    "learned ahead"
                    if low > 0.0
                    else ("baseline ahead" if high < 0.0 else "not distinguished")
                )
                lines.append(
                    f"  learned - {baseline:<20s} k={k:<3d} {mean:+.6f} "
                    f"95% CI [{low:+.6f}, {high:+.6f}] {verdict}"
                )
            lines.append(f"{'best at k=' + str(k):<22s} {self.best_at(k)}")
            lines.append("")
        return lines


def ranker_comparison(
    train: Sequence[CorpusEntry],
    test: Sequence[CorpusEntry],
    ks: Sequence[int] = (5, 10, 20),
    drop_features: Sequence[str] = (),
    random_seed: int = 20261024,
    ranker: RiskRanker | None = None,
) -> RankerComparison:
    """Fit a ranker on ``train`` and score four orderings on ``test``.

    Parameters
    ----------
    train, test:
        Corpus entries. The caller is responsible for generating them from
        disjoint seed ranges; :func:`cmdguard.benchmark.assert_disjoint` checks
        it and the validation script calls that check.
    ks:
        Cut-offs for precision at k.
    drop_features:
        Features withheld from the learned model, for ablation. The timing
        ablation is the one that matters on this corpus and the reason is in
        ``MODEL_CARD.md``.
    random_seed:
        Seed for the random control ordering.
    ranker:
        A pre-built ranker; one is constructed when ``None``.

    Returns
    -------
    RankerComparison
    """
    if not train or not test:
        raise ValueError("ranker_comparison needs a non-empty train and test corpus")
    for k in ks:
        if k < 1:
            raise ValueError(f"k={k} must be >= 1")

    model = ranker if ranker is not None else RiskRanker(drop_features=drop_features)
    model.fit([(entry.case, entry.report) for entry in train])
    assert model.metrics is not None

    orderings: dict[str, object] = {
        "learned": lambda case: model.order(case),
        "irreversibility_flag": irreversibility_ordering,
        "resource_draw": resource_draw_ordering,
        "random_control": lambda case: random_ordering(case, random_seed),
    }

    labels = {
        entry.case.name: involved_mask(entry.report, len(entry.case.sequence)) for entry in test
    }
    positive = np.concatenate([labels[entry.case.name] for entry in test])

    start = time.perf_counter()
    scores: dict[tuple[str, int], OrderingScore] = {}
    for label, builder in orderings.items():
        cached = {entry.case.name: builder(entry.case) for entry in test}  # type: ignore[operator]
        for k in ks:
            values: list[float] = []
            recalls: list[float] = []
            excluded = 0
            for entry in test:
                value = precision_at_k(cached[entry.case.name], labels[entry.case.name], k)
                if value != value:
                    excluded += 1
                else:
                    values.append(value)
                    recalls.append(
                        recall_at_k(cached[entry.case.name], labels[entry.case.name], k)
                    )
            scores[(label, k)] = OrderingScore(
                label=label,
                k=k,
                mean_precision=float(np.mean(values)) if values else float("nan"),
                mean_recall=float(np.mean(recalls)) if recalls else float("nan"),
                n_scored=len(values),
                n_excluded=excluded,
                per_case_precision=tuple(values),
            )
    score_seconds = time.perf_counter() - start

    return RankerComparison(
        ks=tuple(int(k) for k in ks),
        scores=scores,
        n_train_cases=len(train),
        n_test_cases=len(test),
        positive_fraction_test=float(positive.mean()) if positive.size else float("nan"),
        fit_seconds=model.metrics.fit_seconds,
        score_seconds=score_seconds,
        features_dropped=model.drop_features,
        feature_importance=model.metrics.feature_importance,
        mean_involved_per_case=float(
            np.mean([labels[entry.case.name].sum() for entry in test])
        ),
        mean_steps_per_case=float(np.mean([len(entry.case.sequence) for entry in test])),
        bootstrap_seed=random_seed,
    )


def assert_disjoint(train_seeds: Iterable[int], test_seeds: Iterable[int]) -> tuple[int, int]:
    """Raise when the train and test seed ranges overlap.

    Returns
    -------
    tuple
        ``(n_train_seeds, n_test_seeds)``.

    Raises
    ------
    ValueError
        On any overlap. A held-out score measured on a seed the model trained
        on is not a held-out score, and this is one line of checking.
    """
    left = set(train_seeds)
    right = set(test_seeds)
    overlap = sorted(left & right)
    if overlap:
        raise ValueError(
            f"train and test seed ranges overlap on {len(overlap)} seed(s): "
            f"{overlap[:8]}{' ...' if len(overlap) > 8 else ''}"
        )
    return len(left), len(right)


def timing_table(cases: Sequence[Case], repeats: int = 3) -> list[tuple[str, int, float, float]]:
    """Wall-clock cost of forward execution and of the lint, per case.

    Returns ``(case name, steps, forward ms, lint ms)`` with the mean over
    ``repeats``. These numbers belong in a table explicitly declared unstable:
    they are wall clock on a shared two-core container and they move between
    runs. No prose conclusion in this repository is derived from them.
    """
    from .trace import forward_execute

    if repeats < 1:
        raise ValueError(f"repeats={repeats} must be >= 1")
    rows: list[tuple[str, int, float, float]] = []
    for case in cases:
        forward_ms = 0.0
        lint_ms = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            forward_execute(case)
            forward_ms += (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            static_lint(case, LintStrength.WITH_TIMING)
            lint_ms += (time.perf_counter() - start) * 1000.0
        rows.append((case.name, len(case.sequence), forward_ms / repeats, lint_ms / repeats))
    return rows


def unused_report_guard(report: CheckReport) -> int:
    """Number of violations in ``report``. Kept so the import is not decorative."""
    return report.n_violations
