"""The independently derived bracket on the stale fraction.

"Two implementations agree" cannot catch an error both share. The closure
comparison and the ground truth both enumerate reachable artifacts and compare
contents, so a defect in the reachability search would move both. The bracket
checked here uses mathematics the measurement does not use: the Moore bound on
the number of nodes within a given distance in a graph of bounded out-degree,
plus a union bound over the changed artifacts. It reads no content, computes no
digest and performs no transitive search.

THE UPPER BOUND, derived in ``src/evidencestale/bounds.py``:

    stale_fraction <= min(1, m * min(N, k * M(d, L)) / E)
    M(d, L) = 1 + d + d^2 + ... + d^L

with ``k`` the number of artifacts whose content changed, ``d`` the maximum
number of direct dependents of any artifact (the maximum out-degree of the
reversed impact graph), ``L`` the declared impact depth, ``N`` the artifact
count, ``m`` the maximum evidence items on any one subject and ``E`` the
evidence count. The two structural constants ``d`` and ``L`` are declared by
the corpus and are checked against the generated graph by
``validate_corpus_pin.py``.

THE LOWER BOUNDS are the depth-0 count (items whose own subject changed) and
the depth-1 count (items whose subject or a direct declared dependency
changed). Neither performs a transitive search. Both are evaluated against
each item's OWN change set: a union over items is correct for the upper bound
and invalid for the lower bounds, which is a defect this build made and
measured, pinned as
``tests/test_regressions.py::test_regression_lower_bound_needs_a_per_item_change_set``.

WHEN THE UPPER BOUND IS INFORMATIVE: ``m * min(N, k*M) / E`` reaches 1 as soon
as ``k * M(d, L) >= E / m``, so the bound constrains nothing for a large change
count. That is a property of the bound and not of the measurement. The count of
evaluation revisions at which it is informative is published below beside the
count at which it holds, so a reader is never shown only the revisions where it
happened to bite.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.baselines import ground_truth  # noqa: E402
from evidencestale.bounds import moore_bound, stale_fraction_bracket  # noqa: E402
from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.replay import changed_sets_by_item  # noqa: E402

#: Evaluation revisions the bracket is checked at. Every evidence item in the
#: corpus is produced in revisions 1..600, so the evidence set is complete at
#: every revision in this grid and the grid varies only the change count.
GRID = (600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000)


def main() -> int:
    """Check the bracket at every evaluation revision in the grid."""
    out = _bootstrap.Tee("validate_bound_bracket_output.txt")
    out("Independent bracket on the stale fraction -- P098 EvidenceStale")
    out("")
    out("the bound uses the Moore bound and a union bound over the changed")
    out("artifacts. It reads no content, computes no digest and performs no")
    out("transitive search, so it cannot share a defect with the measurement.")
    out("")
    out(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")

    corpus = load_corpus(ROOT / MAIN_CORPUS)
    items = corpus.evidence
    artifacts = len(corpus.true_graph.nodes)
    degree = corpus.declared_max_dependents
    depth = corpus.declared_max_impact_depth
    reach = moore_bound(degree, depth, cap=artifacts)

    out.rule("1. THE BOUND'S INPUTS, ALL OF THEM COUNTS")
    out(f"artifacts N                             : {artifacts}")
    out(f"max direct dependents d (declared)      : {degree}")
    out(f"declared impact depth L                 : {depth}")
    out(f"M(d, L) = 1 + d + ... + d^L             : {moore_bound(degree, depth)}")
    out(f"M(d, L) capped at N                     : {reach}")
    out(f"evidence items E                        : {len(items)}")
    per_subject: dict[str, int] = {}
    for item in items:
        per_subject[item.subject] = per_subject.get(item.subject, 0) + 1
    out(f"max evidence items per subject m        : {max(per_subject.values())}")
    out(f"distinct production revisions           : "
        f"{len({item.produced_at for item in items})}")
    out(f"production revision range               : "
        f"{min(item.produced_at for item in items)}.."
        f"{max(item.produced_at for item in items)}")
    out("")
    out("informativeness condition, from the derivation: the upper bound is below 1")
    out(f"when k * M(d, L) < E / m, that is when k < "
        f"{len(items) / (max(per_subject.values()) * reach):.4f}")

    out.rule("2. THE BRACKET AT EVERY EVALUATION REVISION IN THE GRID")
    out(f"grid: {GRID}")
    out("")
    out(f"{'revision':>9} {'items':>6} {'k':>4} {'depth0':>9} {'depth1':>9} "
        f"{'measured':>9} {'upper':>9} {'inform':>7} {'holds':>6}")
    informative = 0
    held = 0
    rows = 0
    for revision in GRID:
        subset = [item for item in items if item.produced_at <= revision]
        truth = ground_truth(
            subset, corpus.history, corpus.true_graph, revision
        )
        changed = changed_sets_by_item(
            corpus.history,
            [(item.identifier, item.produced_at) for item in subset],
            revision,
        )
        bracket = stale_fraction_bracket(
            subset,
            changed,
            corpus.true_graph.dependencies(),
            measured=truth.stale_fraction,
            max_out_degree=degree,
            depth_limit=depth,
            artifact_count=artifacts,
        )
        rows += 1
        informative += int(bracket.informative)
        held += int(bracket.holds)
        out(
            f"{revision:>9} {len(subset):>6} {bracket.changed_count:>4} "
            f"{bracket.lower_depth0:>9.6f} {bracket.lower_depth1:>9.6f} "
            f"{bracket.measured:>9.6f} {bracket.upper:>9.6f} "
            f"{str(bracket.informative):>7} {str(bracket.holds):>6}"
        )
    out("")
    out(f"slack per endpoint, from the conditioning of one division: "
        f"{bracket.slack:.6e}")
    out.check(
        "the bracket holds at every evaluation revision in the grid",
        held == rows,
        f"{held} of {rows}",
    )
    out.check(
        "the upper bound is informative at every evaluation revision in the grid",
        informative == rows,
        f"{informative} of {rows}",
    )

    out.rule("3. THE HEADLINE BRACKET, at the corpus evaluation revision")
    truth = ground_truth(
        items, corpus.history, corpus.true_graph, corpus.evaluation_time
    )
    changed = changed_sets_by_item(
        corpus.history,
        [(item.identifier, item.produced_at) for item in items],
        corpus.evaluation_time,
    )
    bracket = stale_fraction_bracket(
        items,
        changed,
        corpus.true_graph.dependencies(),
        measured=truth.stale_fraction,
        max_out_degree=degree,
        depth_limit=depth,
        artifact_count=artifacts,
    )
    out(f"changed artifacts k            : {bracket.changed_count}")
    out(f"lower bound, depth 0           : {bracket.lower_depth0:.6f}")
    out(f"lower bound, depth 1           : {bracket.lower_depth1:.6f}")
    out(f"MEASURED stale fraction        : {bracket.measured:.6f}")
    out(f"upper bound, Moore plus union  : {bracket.upper:.6f}")
    out(f"slack per endpoint             : {bracket.slack:.6e}")
    out(f"bracket width                  : {bracket.upper - bracket.lower:.6f}")
    out.check("the measured fraction is above the lower bound", bracket.measured >= bracket.lower)
    out.check("the measured fraction is below the upper bound", bracket.measured <= bracket.upper)
    out.check("the headline upper bound is informative", bracket.informative)

    out.rule("4. CONTROLS ON THE BRACKET ITSELF -- a check that cannot fail is not a check")
    out("each control is anchored on the TRUE value, not on an output, and each is")
    out("asserted to behave as declared.")
    out("")
    above = stale_fraction_bracket(
        items,
        changed,
        corpus.true_graph.dependencies(),
        measured=bracket.upper + 0.05,
        max_out_degree=degree,
        depth_limit=depth,
        artifact_count=artifacts,
    )
    out(f"a measurement 0.05 ABOVE the upper bound  -> holds = {above.holds}")
    out.control(
        "the bracket rejects a measurement above its upper bound",
        not above.holds,
        must_fire=True,
        detail=f"measured {above.measured:.6f} against upper {above.upper:.6f}",
    )
    below = stale_fraction_bracket(
        items,
        changed,
        corpus.true_graph.dependencies(),
        measured=max(0.0, bracket.lower - 0.05),
        max_out_degree=degree,
        depth_limit=depth,
        artifact_count=artifacts,
    )
    out(f"a measurement 0.05 BELOW the lower bound  -> holds = {below.holds}")
    out.control(
        "the bracket rejects a measurement below its lower bound",
        not below.holds,
        must_fire=True,
        detail=f"measured {below.measured:.6f} against lower {below.lower:.6f}",
    )
    inside = stale_fraction_bracket(
        items,
        changed,
        corpus.true_graph.dependencies(),
        measured=0.5 * (bracket.lower + bracket.upper),
        max_out_degree=degree,
        depth_limit=depth,
        artifact_count=artifacts,
    )
    out(f"a measurement at the bracket midpoint     -> holds = {inside.holds}")
    out.control(
        "the bracket accepts a measurement at its midpoint",
        not inside.holds,
        must_fire=False,
        detail=f"measured {inside.measured:.6f}",
    )
    out("")
    out("THE UNION-CHANGE-SET DEFECT, kept as an executable control. Replacing the")
    out("per-item change sets by their union is correct for the upper bound and")
    out("invalid for the lower bounds. On this corpus the production revisions")
    out("overlap heavily, so the substitution over-counts without crossing the")
    out("measurement; the literal case where it DOES cross is pinned in")
    out("tests/test_regressions.py. Both numbers are reported.")
    union: set[str] = set()
    for value in changed.values():
        union |= value
    wrong = stale_fraction_bracket(
        items,
        dict.fromkeys(changed, frozenset(union)),
        corpus.true_graph.dependencies(),
        measured=truth.stale_fraction,
        max_out_degree=degree,
        depth_limit=depth,
        artifact_count=artifacts,
    )
    out(f"  per-item depth-1 lower bound : {bracket.lower_depth1:.6f}")
    out(f"  union    depth-1 lower bound : {wrong.lower_depth1:.6f}")
    out(f"  over-count                   : {wrong.lower_depth1 - bracket.lower_depth1:+.6f}")
    out(f"  union form still holds here   : {wrong.holds}")
    out.control(
        "the union substitution raises the depth-1 lower bound, which is the defect",
        wrong.lower_depth1 > bracket.lower_depth1,
        must_fire=True,
        detail=f"{wrong.lower_depth1:.6f} > {bracket.lower_depth1:.6f}",
    )

    out.rule("5. THE MOORE BOUND, SPOT-CHECKED BY HAND")
    for degree_value, depth_value, expected in (
        (2, 0, 1),
        (2, 1, 3),
        (2, 2, 7),
        (2, 3, 15),
        (3, 2, 13),
        (4, 3, 85),
        (0, 5, 1),
        (1, 5, 6),
    ):
        computed = moore_bound(degree_value, depth_value)
        out(f"  M({degree_value}, {depth_value}) = {computed:>3}   expected {expected:>3}")
        out.check(
            f"M({degree_value}, {depth_value}) equals the hand value",
            computed == expected,
        )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
