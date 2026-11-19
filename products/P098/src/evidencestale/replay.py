"""Replay of a change history, and the by-construction ground truth.

The corpus this product ships is a replayed change history: a list of
:class:`~evidencestale.model.ChangeEvent` over a fixed artifact set, in which
each evidence item's true staleness is known by construction because the
history records exactly which artifact contents changed and when.

THE DEFINITION OF TRUE STALENESS used throughout, stated once:

    evidence ``e`` about subject ``s``, produced at revision ``tp``, is stale
    at revision ``T`` if and only if there exists an artifact ``a`` in the
    TRUE dependency closure of ``s`` whose content at ``T`` differs from its
    content at ``tp``.

Three consequences, each of which a test exercises:

- A ``touch`` event never makes evidence stale. It bumps the modification
  revision and leaves the content identical, so the definition is unmoved.
  This is the case baseline 1 gets wrong in the false-positive direction.
- A ``write`` that restores an earlier content value does **not** make evidence
  stale, because the definition compares values at two revisions and not the
  number of events between them. This is the case an event-counting method
  gets wrong, and the corpus contains such reverts deliberately.
- Staleness reached through a dependency of a dependency is staleness. This is
  the case baseline 1 gets wrong in the false-negative direction.

The definition is value-based, and that is a modelling choice with a cost: a
change that happens to produce byte-identical content is not a change here. For
an audit that is the right answer -- the evidence still describes the artifact
it describes -- and it is the reason a touched-but-unchanged file is counted as
a baseline 1 false positive rather than as a true positive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .graph import DEFAULT_DEPTH_LIMIT, DependencyGraph
from .model import ChangeEvent


class ChangeHistory:
    """An ordered change history with content and modification-revision lookup.

    Parameters
    ----------
    artifacts:
        Every artifact identifier. Each must have an initial content.
    initial:
        ``artifact -> content`` at revision 0.
    events:
        Change events. Sorted internally by ``(time, artifact, kind)``, so the
        caller need not pre-sort; two events on the same artifact at the same
        revision are a ``ValueError``, because their order would decide the
        answer and the history would not be a replay.
    """

    def __init__(
        self,
        artifacts: Iterable[str],
        initial: Mapping[str, str],
        events: Sequence[ChangeEvent],
    ) -> None:
        self.artifacts: tuple[str, ...] = tuple(artifacts)
        known = set(self.artifacts)
        missing = sorted(known - set(initial))
        if missing:
            raise ValueError(
                f"{len(missing)} artifacts have no initial content: {missing[:8]}. "
                f"Every artifact needs a content at revision 0, because a closure "
                f"comparison at any later revision reads it."
            )
        extra = sorted(set(initial) - known)
        if extra:
            raise ValueError(
                f"initial content was given for {len(extra)} undeclared artifacts: "
                f"{extra[:8]}. Declare them as artifacts or drop them."
            )
        self.initial: dict[str, str] = dict(initial)
        ordered = sorted(events, key=lambda e: (e.time, e.artifact, e.kind))
        seen: set[tuple[int, str]] = set()
        for event in ordered:
            if event.artifact not in known:
                raise ValueError(
                    f"event at revision {event.time} names undeclared artifact "
                    f"{event.artifact!r}. Declare it as an artifact or drop the event."
                )
            key = (event.time, event.artifact)
            if key in seen:
                raise ValueError(
                    f"two events apply to artifact {event.artifact!r} at the same "
                    f"revision {event.time}; their order would decide the replayed "
                    f"content, so the history is not a replay. Give them distinct "
                    f"revision indices."
                )
            seen.add(key)
        self.events: tuple[ChangeEvent, ...] = tuple(ordered)

    # -- state -------------------------------------------------------------

    def contents_at(self, time: int) -> dict[str, str]:
        """``artifact -> content`` after every event with ``event.time <= time``."""
        state = dict(self.initial)
        for event in self.events:
            if event.time > time:
                break
            if event.kind == "write" and event.content is not None:
                state[event.artifact] = event.content
        return state

    def modified_at(self, time: int) -> dict[str, int]:
        """``artifact -> revision of its last event at or before ``time``.

        Artifacts with no event yet report revision 0, which is when their
        initial content was laid down. This is the modification revision
        baseline 1 compares against, and it is bumped by a ``touch``.
        """
        state = dict.fromkeys(self.artifacts, 0)
        for event in self.events:
            if event.time > time:
                break
            state[event.artifact] = event.time
        return state

    def changed_between(self, start: int, end: int) -> frozenset[str]:
        """Artifacts whose CONTENT differs between revisions ``start`` and ``end``.

        Value-based, so a revert to an earlier value is not a change and a
        ``touch`` is not a change. ``O(V + E_h)`` for a history of ``E_h``
        events.
        """
        if end < start:
            raise ValueError(
                f"changed_between(start={start}, end={end}) was called with end < "
                f"start; the comparison is between an earlier and a later revision."
            )
        before = self.contents_at(start)
        after = self.contents_at(end)
        return frozenset(
            name for name in self.artifacts if before.get(name) != after.get(name)
        )

    def write_count_before(self, time: int) -> dict[str, int]:
        """``artifact -> number of ``write`` events strictly before ``time``.

        Historical churn, which is a feature of the learned model and is
        available at audit time without any post-production information.
        """
        counts = dict.fromkeys(self.artifacts, 0)
        for event in self.events:
            if event.time >= time:
                break
            if event.kind == "write":
                counts[event.artifact] += 1
        return counts

    def event_count_before(self, time: int) -> dict[str, int]:
        """``artifact -> number of events of EITHER kind strictly before ``time``."""
        counts = dict.fromkeys(self.artifacts, 0)
        for event in self.events:
            if event.time >= time:
                break
            counts[event.artifact] += 1
        return counts


def true_stale(
    history: ChangeHistory,
    graph: DependencyGraph,
    subject: str,
    produced_at: int,
    evaluation_time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> bool:
    """Ground-truth staleness of evidence about ``subject``, from the history.

    This is baseline 3 for one item: the truth, computed by replaying the
    history over the TRUE dependency graph. It takes the true graph, never a
    declared subset, which is what makes it the reference the other two
    baselines are scored against.
    """
    if evaluation_time < produced_at:
        raise ValueError(
            f"evaluation_time={evaluation_time} precedes produced_at={produced_at}; "
            f"evidence cannot be evaluated before it was produced."
        )
    closure = graph.closure(subject, depth_limit=depth_limit)
    changed = history.changed_between(produced_at, evaluation_time)
    return bool(closure & changed)


def true_stale_labels(
    history: ChangeHistory,
    graph: DependencyGraph,
    items: Sequence[tuple[str, str, int]],
    evaluation_time: int,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> dict[str, bool]:
    """Ground truth for many items, as ``identifier -> stale``.

    ``items`` are ``(identifier, subject, produced_at)`` triples. The changed
    set is computed once per distinct ``produced_at`` rather than once per
    item, which is the whole saving: ``O(D * (V + E_h) + n * |closure|)`` for
    ``D`` distinct production revisions.
    """
    closures = {
        subject: graph.closure(subject, depth_limit=depth_limit)
        for _identifier, subject, _produced in items
    }
    changed_cache: dict[int, frozenset[str]] = {}
    out: dict[str, bool] = {}
    for identifier, subject, produced in items:
        if produced not in changed_cache:
            changed_cache[produced] = history.changed_between(produced, evaluation_time)
        out[identifier] = bool(closures[subject] & changed_cache[produced])
    return out


def changed_sets_by_item(
    history: ChangeHistory,
    items: Sequence[tuple[str, int]],
    evaluation_time: int,
) -> dict[str, frozenset[str]]:
    """``identifier -> artifacts whose content changed in that item's own window``.

    ``items`` are ``(identifier, produced_at)`` pairs. The result is what
    ``evidencestale.bounds.stale_fraction_bracket`` needs: a change set per
    item, because a union over items is valid for its upper bound and invalid
    for its lower bounds. The sets are cached by production revision, so the
    cost is ``O(D * (V + E_h))`` for ``D`` distinct production revisions rather
    than once per item.
    """
    cache: dict[int, frozenset[str]] = {}
    out: dict[str, frozenset[str]] = {}
    for identifier, produced in items:
        if produced not in cache:
            cache[produced] = history.changed_between(produced, evaluation_time)
        out[identifier] = cache[produced]
    return out
