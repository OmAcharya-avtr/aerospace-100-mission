"""The three baselines, in the order the build guide requires them implemented.

BASELINE 1 -- TIMESTAMP COMPARISON, THE THING BEING REFUTED
-----------------------------------------------------------
:func:`baseline_timestamp` declares evidence stale when the modification
revision of its subject artifact is later than the revision at which the
evidence was produced. This is what a traceability matrix with a date column
does, and it is what this product exists to measure the error of.

It is implemented at full strength and not weakened to lose: the comparison is
strict (``mtime > produced_at``, so an artifact modified at the same revision as
the evidence was produced is not called stale, which is the correct reading of
"the evidence was produced from that revision"), the modification revision is
the real last-event revision from the replayed history, and
:func:`baseline_timestamp_depth1` is also provided -- the strongest honest
variant of the same idea, which takes the maximum modification revision over
the subject and its DIRECT declared dependencies. Both are reported. The
depth-1 variant still cannot see a change that reaches the artifact at depth 2
or more, which is the structural failure, not an implementation shortfall.

Its two error directions are different failures with different audit
consequences and are never summed:

- **Missed staleness (false negative).** A dependency of the subject changed,
  the subject's own file was not touched, so the timestamp comparison reports
  the evidence as current. An audit that trusts it cites evidence that no
  longer describes the system. This is the dangerous direction.
- **False alarm (false positive).** The subject was touched -- rebuilt,
  reformatted, re-checked-out -- with identical content, so the timestamp
  comparison reports the evidence as stale. An audit that trusts it re-runs
  work that did not need re-running, and, worse, learns to discount the tool.

BASELINE 2 -- CONTENT-HASH COMPARISON OVER THE DEPENDENCY CLOSURE
-----------------------------------------------------------------
:func:`baseline_closure_hash` compares the SHA-256 digest of every artifact in
the subject's declared dependency closure against the digest recorded when the
evidence was produced. Under the conditions listed in
``validation/VALIDATION.md`` it agrees with the ground truth on every item; it
is exact in exact arithmetic and the comparison is over integers and byte
strings, so there is no rounding to agree to. The conditions are: complete
dependency declarations, no digest collision, and the closure depth limit not
binding. Each is measured and published rather than assumed.

BASELINE 3 -- THE REPLAYED CHANGE HISTORY, WHICH IS THE TRUTH
-------------------------------------------------------------
:func:`ground_truth` returns the by-construction label from
``evidencestale.replay``, over the TRUE dependency graph. Aggregate intervals
come from ``evidencestale.bootstrap``, whose resampling unit is declared and
asserted there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .graph import DEFAULT_DEPTH_LIMIT, DependencyGraph
from .hashing import content_hash
from .model import EvidenceItem
from .replay import ChangeHistory, true_stale_labels


@dataclass(frozen=True)
class StalenessVerdicts:
    """Per-item staleness verdicts from one method, plus the aggregate fraction.

    Attributes
    ----------
    method:
        Short method name, used in reports.
    verdicts:
        ``evidence identifier -> stale``.
    order:
        Evidence identifiers in the order they were supplied, so the caller can
        line two methods' verdicts up without relying on mapping order.
    """

    method: str
    verdicts: Mapping[str, bool]
    order: tuple[str, ...]

    @property
    def stale_count(self) -> int:
        """Number of items called stale."""
        return sum(1 for name in self.order if self.verdicts[name])

    @property
    def total(self) -> int:
        """Number of items judged."""
        return len(self.order)

    @property
    def stale_fraction(self) -> float:
        """Headline output: stale items divided by items judged, dimensionless.

        ``0.0`` for an empty item set, which is reported rather than raised
        because an assurance case with no evidence has nothing stale in it; the
        caller sees ``total == 0``.
        """
        if not self.order:
            return 0.0
        return self.stale_count / len(self.order)

    def as_array(self) -> list[int]:
        """Verdicts as 0/1 ints in :attr:`order`, for the metric functions."""
        return [int(self.verdicts[name]) for name in self.order]


def _check_items(items: Sequence[EvidenceItem], graph: DependencyGraph) -> None:
    known = set(graph.nodes)
    identifiers: set[str] = set()
    for item in items:
        if item.subject not in known:
            raise ValueError(
                f"evidence {item.identifier!r} is about artifact {item.subject!r}, "
                f"which is not a declared artifact in the dependency graph. An "
                f"evidence item whose subject is undeclared has no closure, so its "
                f"staleness is undefined; declare the artifact or drop the item."
            )
        if item.identifier in identifiers:
            raise ValueError(
                f"evidence identifier {item.identifier!r} appears more than once; "
                f"identifiers index the verdict mapping and must be unique."
            )
        identifiers.add(item.identifier)


def baseline_timestamp(
    items: Sequence[EvidenceItem],
    history: ChangeHistory,
    graph: DependencyGraph,
    evaluation_time: int,
) -> StalenessVerdicts:
    """Baseline 1: the subject's modification revision against the evidence date.

    ``stale  <=>  modified_at(evaluation_time)[subject] > produced_at``.

    Parameters
    ----------
    items:
        Evidence items to judge.
    history:
        The replayed change history supplying modification revisions.
    graph:
        Only used to validate that each subject is a declared artifact.
    evaluation_time:
        Revision at which the audit is performed.

    Returns
    -------
    StalenessVerdicts
    """
    _check_items(items, graph)
    modified = history.modified_at(evaluation_time)
    verdicts = {
        item.identifier: modified[item.subject] > item.produced_at for item in items
    }
    return StalenessVerdicts(
        method="baseline1_timestamp",
        verdicts=verdicts,
        order=tuple(item.identifier for item in items),
    )


def baseline_timestamp_depth1(
    items: Sequence[EvidenceItem],
    history: ChangeHistory,
    graph: DependencyGraph,
    evaluation_time: int,
) -> StalenessVerdicts:
    """Baseline 1 strengthened to the subject's DIRECT declared dependencies.

    The strongest honest form of the timestamp idea: it reads one level of the
    declared graph. It still cannot see a change at depth 2 or more, which is
    the structural limit of comparing modification revisions.
    """
    _check_items(items, graph)
    modified = history.modified_at(evaluation_time)
    dependencies = graph.dependencies()
    verdicts: dict[str, bool] = {}
    for item in items:
        neighbourhood = (item.subject, *dependencies[item.subject])
        latest = max(modified[name] for name in neighbourhood)
        verdicts[item.identifier] = latest > item.produced_at
    return StalenessVerdicts(
        method="baseline1b_timestamp_depth1",
        verdicts=verdicts,
        order=tuple(item.identifier for item in items),
    )


def closure_snapshot(
    history: ChangeHistory,
    graph: DependencyGraph,
    subject: str,
    time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> dict[str, str]:
    """``artifact -> SHA-256 digest`` over the declared closure of ``subject``.

    This is the record an evidence item carries forward. Computed from the
    replayed history at revision ``time``.
    """
    closure = graph.closure(subject, depth_limit=depth_limit)
    contents = history.contents_at(time)
    return {name: content_hash(contents[name]) for name in sorted(closure)}


def baseline_closure_hash(
    items: Sequence[EvidenceItem],
    history: ChangeHistory,
    graph: DependencyGraph,
    evaluation_time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> StalenessVerdicts:
    """Baseline 2: digest comparison over the subject's declared closure.

    ``stale  <=>  there is an artifact in closure(subject) whose digest at
    evaluation_time differs from its digest at produced_at``.

    Exact against the ground truth when the declared graph is the true graph,
    no two contents share a digest and the depth limit does not bind. Those
    three conditions are measured in ``validation/validate_baselines.py`` and
    none is assumed here.
    """
    _check_items(items, graph)
    closures = {
        item.subject: graph.closure(item.subject, depth_limit=depth_limit)
        for item in items
    }
    current_contents = history.contents_at(evaluation_time)
    produced_contents: dict[int, dict[str, str]] = {}
    digest_cache: dict[str, str] = {}

    def digest(value: str) -> str:
        if value not in digest_cache:
            digest_cache[value] = content_hash(value)
        return digest_cache[value]

    verdicts: dict[str, bool] = {}
    for item in items:
        if item.produced_at not in produced_contents:
            produced_contents[item.produced_at] = history.contents_at(item.produced_at)
        before = produced_contents[item.produced_at]
        stale = False
        for name in closures[item.subject]:
            if digest(before[name]) != digest(current_contents[name]):
                stale = True
                break
        verdicts[item.identifier] = stale
    return StalenessVerdicts(
        method="baseline2_closure_hash",
        verdicts=verdicts,
        order=tuple(item.identifier for item in items),
    )


def ground_truth(
    items: Sequence[EvidenceItem],
    history: ChangeHistory,
    true_graph: DependencyGraph,
    evaluation_time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> StalenessVerdicts:
    """Baseline 3: the by-construction truth, over the TRUE dependency graph."""
    _check_items(items, true_graph)
    labels = true_stale_labels(
        history,
        true_graph,
        [(item.identifier, item.subject, item.produced_at) for item in items],
        evaluation_time,
        depth_limit=depth_limit,
    )
    return StalenessVerdicts(
        method="baseline3_ground_truth",
        verdicts=labels,
        order=tuple(item.identifier for item in items),
    )


def committed_labels(items: Sequence[EvidenceItem]) -> StalenessVerdicts:
    """The ground-truth labels carried in the corpus, as written by the generator.

    These are the labels the corpus generator computed while it built the
    history, before any code in this package existed to read them. Comparing
    them against :func:`ground_truth` checks the replay against the generator;
    comparing them against :func:`baseline_closure_hash` is the exactness
    measurement.
    """
    missing = [item.identifier for item in items if item.true_stale is None]
    if missing:
        raise ValueError(
            f"{len(missing)} evidence items carry no by-construction label "
            f"(true_stale is None), beginning {missing[:8]}. The committed-label "
            f"comparison needs every item labelled; load a corpus that carries "
            f"labels, or score against ground_truth() instead."
        )
    return StalenessVerdicts(
        method="corpus_committed_labels",
        verdicts={item.identifier: bool(item.true_stale) for item in items},
        order=tuple(item.identifier for item in items),
    )
