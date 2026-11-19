"""An independent bracket on the stale fraction, from counting arguments only.

WHY THIS MODULE EXISTS
----------------------
"Two implementations agree" cannot catch an error both share. The closure
comparison and the ground truth both enumerate reachable artifacts and compare
contents; a defect in the reachability search would move both. So this module
brackets the stale fraction using mathematics the measurement does not use: the
Moore bound on the number of nodes within a given distance in a graph of
bounded out-degree, and a union bound over the changed artifacts. It consumes
four integers -- the change count, the maximum out-degree of the impact graph,
the depth limit and the artifact count -- plus two counts from the evidence
set. It performs no reachability search, reads no content and computes no
digest.

THE UPPER BOUND
---------------
Let ``C`` be the set of artifacts whose content changed in the interval, with
``k = |C|``. Let ``d`` be the maximum number of direct dependents of any
artifact, which is the maximum out-degree of the reversed ("impact") graph. Let
``L`` be the depth limit and ``N`` the number of artifacts.

An evidence item is stale only if its subject can reach a member of ``C``, that
is only if its subject lies in the impact set of ``C``. From any one seed, the
number of nodes within distance ``L`` in a graph of out-degree at most ``d`` is
at most

    M(d, L) = 1 + d + d^2 + ... + d^L

which is the Moore bound (standard; see for instance B. Bollobas, *Modern Graph
Theory*, Springer, 1998, for the degree-diameter counting argument). A union
bound over the ``k`` seeds gives ``|impact(C)| <= min(N, k * M(d, L))``. If no
artifact carries more than ``m`` evidence items, the number of stale evidence
items is at most ``m * min(N, k * M(d, L))``, so with ``E`` evidence items

    stale_fraction <= min(1, m * min(N, k * M(d, L)) / E).

THE LOWER BOUNDS, AND A TRAP THIS BUILD WALKED INTO FIRST
---------------------------------------------------------
Two lower bounds, neither of which performs a transitive search:

- **depth 0.** An evidence item whose own subject changed is stale. Counting
  the items whose subject changed is a lower bound and needs no graph at all.
- **depth 1.** An evidence item one of whose subject's DIRECT declared
  dependencies changed is stale. One adjacency lookup, no transitive closure.

The depth-1 count is the larger of the two and is the reported lower bound; the
depth-0 count is reported beside it.

**Each lower bound must be evaluated against the item's OWN change set, not
against a union over items.** The first version of this module took one changed
set for the whole evidence set, the union over items of the artifacts that
changed between that item's production revision and the evaluation revision.
That union is correct for the UPPER bound, because ``impact`` is monotone in
the changed set, and it is wrong for the lower bounds: an artifact that changed
in some other item's window need not have changed in this item's window, so
"this item's subject is in the union" does not imply this item is stale. The
measured consequence, at the corpus and grid this product ships, was a lower
bound above the measured fraction at 6 of 10 evaluation revisions -- a check
that reads as a refutation of the measurement when it is a defect in the check.
So :func:`stale_fraction_bracket` takes a change set **per evidence item** and
forms the union itself for the upper bound.

WHEN THE UPPER BOUND IS INFORMATIVE
-----------------------------------
``m * min(N, k * M(d, L)) / E`` exceeds 1 as soon as ``k * M(d, L)`` saturates
at ``N`` and ``m * N >= E``. A bound equal to 1 is still a true bound and is
still checked, but it constrains nothing, so :func:`stale_fraction_bracket`
reports ``informative`` separately from ``holds``. The validation script
publishes the count of evaluation revisions at which the upper bound is
informative, rather than quoting only the ones where it is.

NUMERICAL CONDITIONING OF THE SLACK
-----------------------------------
Every quantity above is an exact Python integer up to the final division by
``E``. The bracket therefore has exactly two floating-point operations in it,
one division per endpoint, each with relative condition number 1. The slack is
``4 * eps * max(1, |endpoint|)``: four units in the last place at the magnitude
of the endpoint being compared, which follows the conditioning of the division
rather than an absolute constant. An absolute slack would be wrong in both
directions -- vacuous for endpoints near 1 and still arbitrary for endpoints
near 0.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .model import EvidenceItem

#: Machine epsilon for IEEE 754 binary64, from the interpreter rather than a
#: literal, so the slack cannot drift from the arithmetic it describes.
EPSILON = sys.float_info.epsilon

#: Units in the last place of slack allowed on each endpoint. Two divisions in
#: the bracket, each of relative condition number 1, doubled for the comparison.
SLACK_ULPS = 4.0


def moore_bound(out_degree: int, depth: int, cap: int | None = None) -> int:
    """``1 + d + d^2 + ... + d^depth``, optionally capped at ``cap``.

    Computed with Python integers and short-circuited at ``cap``, because the
    uncapped sum overflows any fixed width for the depth limits this product
    uses and the capped value is the only one the bound needs.
    """
    if isinstance(out_degree, bool) or not isinstance(out_degree, int):
        raise TypeError(f"out_degree must be an int, got {type(out_degree).__name__}")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError(f"depth must be an int, got {type(depth).__name__}")
    if out_degree < 0:
        raise ValueError(f"out_degree must be non-negative, got {out_degree}")
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}")
    if cap is not None and cap < 0:
        raise ValueError(f"cap must be non-negative when given, got {cap}")
    total = 1
    term = 1
    for _level in range(depth):
        term *= out_degree
        total += term
        if cap is not None and total >= cap:
            return cap
        if term == 0:
            break
    return total if cap is None else min(total, cap)


@dataclass(frozen=True)
class StaleFractionBracket:
    """An independently derived bracket on the stale fraction.

    Attributes
    ----------
    lower_depth0, lower_depth1:
        Lower bounds from the items whose subject, respectively whose subject or
        a direct declared dependency, is in the changed set.
    upper:
        The Moore-plus-union upper bound, clipped at 1.
    measured:
        The fraction being bracketed, supplied by the caller.
    changed_count, max_out_degree, depth_limit, artifact_count:
        The four integers the upper bound consumes.
    max_items_per_subject, evidence_count:
        The two evidence-set counts the upper bound consumes.
    slack:
        Floating-point slack allowed on each endpoint comparison.
    """

    lower_depth0: float
    lower_depth1: float
    upper: float
    measured: float
    changed_count: int
    max_out_degree: int
    depth_limit: int
    artifact_count: int
    max_items_per_subject: int
    evidence_count: int
    slack: float

    @property
    def lower(self) -> float:
        """The reported lower bound: the larger of the two."""
        return max(self.lower_depth0, self.lower_depth1)

    @property
    def informative(self) -> bool:
        """Whether the upper bound constrains anything, i.e. is below 1."""
        return self.upper < 1.0 - self.slack

    @property
    def holds(self) -> bool:
        """Whether the measured fraction lies inside the bracket, within slack."""
        return (
            self.measured >= self.lower - self.slack
            and self.measured <= self.upper + self.slack
        )


def stale_fraction_bracket(
    items: Sequence[EvidenceItem],
    changed_by_item: Mapping[str, Iterable[str]],
    dependencies: Mapping[str, Sequence[str]],
    *,
    measured: float,
    max_out_degree: int,
    depth_limit: int,
    artifact_count: int,
) -> StaleFractionBracket:
    """Bracket the stale fraction without any reachability search.

    Parameters
    ----------
    items:
        The evidence items the measured fraction was computed over.
    changed_by_item:
        ``evidence identifier -> artifacts whose CONTENT changed between that
        item's production revision and the evaluation revision``. Per item, not
        a union: see the module docstring for the measured consequence of
        getting this wrong. The union is formed here for the upper bound.
    dependencies:
        ``artifact -> direct declared dependencies``. Used only for the depth-1
        lower bound, which is one adjacency lookup and not a closure.
    measured:
        The stale fraction to bracket, dimensionless.
    max_out_degree:
        Maximum number of direct dependents of any artifact, a degree count.
    depth_limit:
        The depth limit ``L`` in the Moore bound.
    artifact_count:
        ``N``, the number of artifacts.

    Returns
    -------
    StaleFractionBracket
    """
    if not items:
        raise ValueError(
            "stale_fraction_bracket() needs at least one evidence item; the bracket is "
            "a statement about a fraction over a non-empty evidence set."
        )
    if artifact_count < 1:
        raise ValueError(f"artifact_count must be at least 1, got {artifact_count}")
    missing = [item.identifier for item in items if item.identifier not in changed_by_item]
    if missing:
        raise ValueError(
            f"{len(missing)} evidence items have no change set in changed_by_item, "
            f"beginning {missing[:8]}. Each item needs the artifacts that changed "
            f"between ITS OWN production revision and the evaluation revision; a union "
            f"over items is correct for the upper bound and invalid for the lower "
            f"bounds. Build it with replay.changed_sets_by_item()."
        )
    per_item = {
        item.identifier: frozenset(changed_by_item[item.identifier]) for item in items
    }
    changed_set: set[str] = set()
    for value in per_item.values():
        changed_set |= value
    total = len(items)

    per_subject: dict[str, int] = {}
    for item in items:
        per_subject[item.subject] = per_subject.get(item.subject, 0) + 1
    max_items_per_subject = max(per_subject.values())

    depth0 = sum(1 for item in items if item.subject in per_item[item.identifier])
    depth1 = 0
    for item in items:
        neighbourhood = {item.subject, *dependencies.get(item.subject, ())}
        if neighbourhood & per_item[item.identifier]:
            depth1 += 1

    reach = moore_bound(max_out_degree, depth_limit, cap=artifact_count)
    impacted_at_most = min(artifact_count, len(changed_set) * reach)
    upper_numerator = max_items_per_subject * impacted_at_most
    upper = min(1.0, upper_numerator / total)
    lower0 = depth0 / total
    lower1 = depth1 / total
    slack = SLACK_ULPS * EPSILON * max(1.0, abs(upper), abs(lower1))
    return StaleFractionBracket(
        lower_depth0=lower0,
        lower_depth1=lower1,
        upper=upper,
        measured=float(measured),
        changed_count=len(changed_set),
        max_out_degree=max_out_degree,
        depth_limit=depth_limit,
        artifact_count=artifact_count,
        max_items_per_subject=max_items_per_subject,
        evidence_count=total,
        slack=slack,
    )
