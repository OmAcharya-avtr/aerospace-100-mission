"""Pinned regressions: every defect this build found, with literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. Every counterexample and every defect found during this
build is therefore pinned here as a named test with the inputs written out.

WHAT HYPOTHESIS FOUND, reported honestly: at 3000 examples per property over
graphs of up to 9 nodes and 20 edges, the eight closure identities in
``tests/test_graph_properties.py`` produced **no** falsifying example against
the code as shipped. The two defects pinned below were found by a
hand-calculated known-answer test and by a validation script respectively; the
first is reachable from the declared Hypothesis strategies and the property
that would have caught it is now in the property-test file, which is checked
here by reproducing the literal case.
"""

from __future__ import annotations

import pytest

from evidencestale.bounds import stale_fraction_bracket
from evidencestale.graph import DependencyGraph
from evidencestale.model import EvidenceItem


def test_regression_depth_limit_equal_to_the_closure_depth_is_sufficient() -> None:
    """Off-by-one in the closure depth guard. Found 2026-11-19 by a known-answer test.

    An earlier version of :meth:`DependencyGraph.closure` ran ``depth_limit``
    breadth-first expansions. A closure whose fixed point needs ``d`` levels
    requires ``d + 1`` expansions -- the last one must discover nothing -- so
    the guard fired at ``depth_limit == d`` and reported malformed input for a
    perfectly well-formed graph. ``closure_depth`` meanwhile reported ``d``, so
    the two disagreed about when the limit bound.

    Literal inputs: a two-node chain at ``depth_limit=1`` and a three-node
    chain at ``depth_limit=2``.
    """
    pair = DependencyGraph.build(["n0", "n1"], [("n0", "n1")])
    assert pair.closure_depth("n0") == 1
    assert pair.closure("n0", depth_limit=1) == frozenset({"n0", "n1"})

    chain = DependencyGraph.build(
        ["top", "middle", "bottom"], [("top", "middle"), ("middle", "bottom")]
    )
    assert chain.closure_depth("top") == 2
    assert chain.closure("top", depth_limit=2) == frozenset({"top", "middle", "bottom"})
    # One level short still raises, which is the guard doing its job.
    with pytest.raises(ValueError, match="did not reach a fixed point"):
        chain.closure("top", depth_limit=1)


def test_regression_lower_bound_needs_a_per_item_change_set() -> None:
    """Union change set invalidates the bracket's lower bounds. Found 2026-11-19.

    The first version of ``bounds.stale_fraction_bracket`` took one changed set
    for the whole evidence set -- the union over items of the artifacts that
    changed in each item's own window. The union is correct for the upper
    bound, because the impact set is monotone in the changed set, and wrong for
    the lower bounds. Measured consequence on the corpus and grid in use at the
    time: a lower bound above the measured fraction at 6 of 10 evaluation
    revisions.

    Literal inputs: two items with different production revisions, one artifact
    ``q`` that changed in the first item's window only, declared as a
    dependency of the second item's subject only.
    """
    items = [
        EvidenceItem(identifier="A", subject="nA", produced_at=0, goal="G", solution="SA"),
        EvidenceItem(identifier="B", subject="nB", produced_at=5, goal="G", solution="SB"),
    ]
    dependencies = {"nA": (), "nB": ("q",), "q": ()}
    arguments = {
        "measured": 0.0,
        "max_out_degree": 2,
        "depth_limit": 2,
        "artifact_count": 50,
    }
    correct = stale_fraction_bracket(
        items, {"A": frozenset({"q"}), "B": frozenset()}, dependencies, **arguments
    )
    assert correct.lower_depth1 == 0.0
    assert correct.holds is True

    union = frozenset({"q"})
    wrong = stale_fraction_bracket(
        items, {"A": union, "B": union}, dependencies, **arguments
    )
    assert wrong.lower_depth1 == pytest.approx(0.5, rel=1e-15, abs=0.0)
    assert wrong.holds is False


def test_regression_collision_bound_does_not_underflow_at_corpus_scale() -> None:
    """A documentation defect, found 2026-11-19 by its own known-answer test.

    ``hashing.collision_probability_bound`` was documented as underflowing to
    ``0.0`` for the sizes used here. It does not: ``12 497 500 * 2**-256`` is of
    order 1e-70, far above the smallest normal double. The docstring claimed a
    numerical property the implementation did not have, which would have made a
    reported ``0.0`` unreadable.
    """
    from evidencestale.hashing import collision_probability_bound

    bound = collision_probability_bound(5000)
    assert bound > 0.0
    assert bound < 1e-60
    assert collision_probability_bound(1) == 0.0
