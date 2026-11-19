"""The independent bracket, with the Moore bound worked by hand."""

from __future__ import annotations

import pytest

from evidencestale.bounds import SLACK_ULPS, moore_bound, stale_fraction_bracket
from evidencestale.model import EvidenceItem
from evidencestale.replay import changed_sets_by_item


def test_moore_bound_known_answers() -> None:
    # 1 + d + d^2 + ... + d^L
    # d=2, L=0 -> 1
    # d=2, L=1 -> 1 + 2 = 3
    # d=2, L=2 -> 1 + 2 + 4 = 7
    # d=2, L=3 -> 1 + 2 + 4 + 8 = 15
    # d=3, L=2 -> 1 + 3 + 9 = 13
    # d=4, L=3 -> 1 + 4 + 16 + 64 = 85
    assert moore_bound(2, 0) == 1
    assert moore_bound(2, 1) == 3
    assert moore_bound(2, 2) == 7
    assert moore_bound(2, 3) == 15
    assert moore_bound(3, 2) == 13
    assert moore_bound(4, 3) == 85


def test_moore_bound_degenerate_degrees() -> None:
    # d=0: nothing is reachable beyond the seed, so the bound is 1.
    assert moore_bound(0, 5) == 1
    # d=1: a path, so 1 + 1 + ... + 1 = L + 1.
    assert moore_bound(1, 5) == 6


def test_moore_bound_cap_short_circuits() -> None:
    # 1 + 2 + 4 = 7 exceeds a cap of 5 at the second term, so 5 is returned
    # without computing the rest. The cap matters because an uncapped sum at
    # the default depth limit of 64 is astronomically large.
    assert moore_bound(2, 2, cap=5) == 5
    assert moore_bound(2, 64, cap=240) == 240
    assert moore_bound(2, 1, cap=100) == 3


def test_moore_bound_input_validation() -> None:
    with pytest.raises(ValueError, match="out_degree must be non-negative"):
        moore_bound(-1, 2)
    with pytest.raises(ValueError, match="depth must be non-negative"):
        moore_bound(2, -1)
    with pytest.raises(ValueError, match="cap must be non-negative"):
        moore_bound(2, 2, cap=-1)
    with pytest.raises(TypeError, match="out_degree must be an int"):
        moore_bound(2.0, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="depth must be an int"):
        moore_bound(2, 2.0)  # type: ignore[arg-type]


def test_bracket_known_answer_by_hand() -> None:
    # Four evidence items, two artifacts, so m = 2.
    # N = 10 artifacts, d = 2, L = 2 -> M(2,2) = 7.
    # Changed set per item: items on "r" saw {"q"} change; items on "s" saw
    # nothing. Union k = 1.
    #   upper  = m * min(N, k*M) / E = 2 * min(10, 7) / 4 = 14 / 4 = 3.5 -> clipped to 1
    # Use N = 2 instead to get an unclipped number:
    #   upper  = 2 * min(2, 7) / 4 = 4/4 = 1.0
    # So take m = 1 by giving each item its own subject.
    items = [
        EvidenceItem(identifier=f"e{index}", subject=f"n{index}", produced_at=0,
                     goal="G", solution=f"Sn{index}")
        for index in range(4)
    ]
    changed = {"e0": {"q"}, "e1": set(), "e2": set(), "e3": set()}
    dependencies = {"n0": ("q",), "n1": (), "n2": (), "n3": ()}
    bracket = stale_fraction_bracket(
        items,
        changed,
        dependencies,
        measured=0.25,
        max_out_degree=2,
        depth_limit=2,
        artifact_count=20,
    )
    # m = 1, k = 1, M(2,2) = 7, N = 20, E = 4
    #   upper = 1 * min(20, 7) / 4 = 7/4 = 1.75 -> clipped to 1.0
    assert bracket.max_items_per_subject == 1
    assert bracket.changed_count == 1
    assert bracket.upper == pytest.approx(1.0, rel=0.0, abs=0.0)
    # depth-0 lower: no item's own subject is in its change set -> 0/4 = 0
    assert bracket.lower_depth0 == pytest.approx(0.0, rel=0.0, abs=0.0)
    # depth-1 lower: e0's subject n0 declares q, which changed -> 1/4 = 0.25
    assert bracket.lower_depth1 == pytest.approx(0.25, rel=1e-15, abs=0.0)
    assert bracket.lower == pytest.approx(0.25, rel=1e-15, abs=0.0)
    assert bracket.holds is True
    assert bracket.informative is False


def test_bracket_informative_when_the_change_set_is_small() -> None:
    # 8 items on 8 distinct subjects, m = 1, N = 100, d = 2, L = 2, M = 7,
    # k = 1 -> upper = 1 * min(100, 7) / 8 = 7/8 = 0.875, which is below 1 and
    # so constrains the measurement.
    items = [
        EvidenceItem(identifier=f"e{index}", subject=f"n{index}", produced_at=0,
                     goal="G", solution=f"Sn{index}")
        for index in range(8)
    ]
    changed = {f"e{index}": ({"q"} if index == 0 else set()) for index in range(8)}
    bracket = stale_fraction_bracket(
        items,
        changed,
        {f"n{index}": () for index in range(8)},
        measured=0.125,
        max_out_degree=2,
        depth_limit=2,
        artifact_count=100,
    )
    assert bracket.upper == pytest.approx(0.875, rel=1e-15, abs=0.0)
    assert bracket.informative is True
    assert bracket.holds is True


def test_bracket_detects_a_measurement_above_the_upper_bound() -> None:
    items = [
        EvidenceItem(identifier=f"e{index}", subject=f"n{index}", produced_at=0,
                     goal="G", solution=f"Sn{index}")
        for index in range(8)
    ]
    changed = {f"e{index}": ({"q"} if index == 0 else set()) for index in range(8)}
    bracket = stale_fraction_bracket(
        items,
        changed,
        {f"n{index}": () for index in range(8)},
        measured=0.95,
        max_out_degree=2,
        depth_limit=2,
        artifact_count=100,
    )
    assert bracket.upper == pytest.approx(0.875, rel=1e-15, abs=0.0)
    assert bracket.holds is False


def test_slack_follows_the_conditioning_of_the_division() -> None:
    items = [
        EvidenceItem(identifier="e0", subject="n0", produced_at=0, goal="G", solution="S0")
    ]
    bracket = stale_fraction_bracket(
        items,
        {"e0": set()},
        {"n0": ()},
        measured=0.0,
        max_out_degree=2,
        depth_limit=2,
        artifact_count=100,
    )
    # Two divisions of relative condition number 1, so the slack is
    # SLACK_ULPS * eps * max(1, |endpoint|). With endpoints in [0, 1] the
    # max is 1, so the slack is exactly SLACK_ULPS * eps.
    import sys

    assert bracket.slack == pytest.approx(
        SLACK_ULPS * sys.float_info.epsilon, rel=1e-15, abs=0.0
    )


def test_bracket_requires_a_change_set_for_every_item() -> None:
    items = [
        EvidenceItem(identifier="e0", subject="n0", produced_at=0, goal="G", solution="S0"),
        EvidenceItem(identifier="e1", subject="n1", produced_at=0, goal="G", solution="S1"),
    ]
    with pytest.raises(ValueError, match="changed_sets_by_item"):
        stale_fraction_bracket(
            items,
            {"e0": set()},
            {"n0": (), "n1": ()},
            measured=0.0,
            max_out_degree=2,
            depth_limit=2,
            artifact_count=10,
        )


def test_bracket_rejects_an_empty_item_set() -> None:
    with pytest.raises(ValueError, match="at least one evidence item"):
        stale_fraction_bracket(
            [], {}, {}, measured=0.0, max_out_degree=1, depth_limit=1, artifact_count=1
        )


def test_bracket_rejects_a_zero_artifact_count() -> None:
    items = [
        EvidenceItem(identifier="e0", subject="n0", produced_at=0, goal="G", solution="S0")
    ]
    with pytest.raises(ValueError, match="artifact_count must be at least 1"):
        stale_fraction_bracket(
            items,
            {"e0": set()},
            {"n0": ()},
            measured=0.0,
            max_out_degree=1,
            depth_limit=1,
            artifact_count=0,
        )


def test_bracket_holds_on_the_committed_corpus(layered_corpus) -> None:
    from evidencestale.baselines import ground_truth

    items = layered_corpus.evidence
    truth = ground_truth(
        items,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    changed = changed_sets_by_item(
        layered_corpus.history,
        [(item.identifier, item.produced_at) for item in items],
        layered_corpus.evaluation_time,
    )
    bracket = stale_fraction_bracket(
        items,
        changed,
        layered_corpus.true_graph.dependencies(),
        measured=truth.stale_fraction,
        max_out_degree=layered_corpus.declared_max_dependents,
        depth_limit=layered_corpus.declared_max_impact_depth,
        artifact_count=len(layered_corpus.true_graph.nodes),
    )
    assert bracket.holds is True
    assert bracket.informative is True
    assert bracket.lower < bracket.measured < bracket.upper


def test_a_union_change_set_breaks_the_lower_bound_on_a_hand_case() -> None:
    """The defect this module's docstring records, kept as an executable check.

    Two items with different production revisions. Artifact ``q`` is written at
    revision 2 and never again, so:

    - item A, produced at revision 0, has ``q`` in its own change set but does
      not declare ``q`` as a dependency;
    - item B, produced at revision 5, declares ``q`` but ``q`` did not change
      in ``(5, 10]``, so B's own change set is empty and B is not stale.

    Per item, the depth-1 lower bound is 0/2 = 0, which is correct. Replacing
    the per-item sets by their union puts ``q`` in B's set too, so the bound
    becomes 1/2 = 0.5 against a measured 0.0 -- a lower bound above the
    measurement, which reads as a refutation and is a defect in the check.
    """
    items = [
        EvidenceItem(identifier="A", subject="nA", produced_at=0, goal="G", solution="SA"),
        EvidenceItem(identifier="B", subject="nB", produced_at=5, goal="G", solution="SB"),
    ]
    dependencies = {"nA": (), "nB": ("q",), "q": ()}
    per_item = {"A": frozenset({"q"}), "B": frozenset()}
    correct = stale_fraction_bracket(
        items,
        per_item,
        dependencies,
        measured=0.0,
        max_out_degree=2,
        depth_limit=2,
        artifact_count=50,
    )
    assert correct.lower_depth1 == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert correct.holds is True

    union = frozenset({"q"})
    wrong = stale_fraction_bracket(
        items,
        {"A": union, "B": union},
        dependencies,
        measured=0.0,
        max_out_degree=2,
        depth_limit=2,
        artifact_count=50,
    )
    assert wrong.lower_depth1 == pytest.approx(0.5, rel=1e-15, abs=0.0)
    assert wrong.holds is False


def test_a_union_change_set_over_counts_on_the_committed_corpus(layered_corpus) -> None:
    """On this corpus the union form over-counts without crossing the measurement.

    Every evidence item here is produced in revisions 1..600 and audited at
    revision 2000, so the per-item windows overlap heavily and the union form's
    error is smaller than on the hand case above. It is still an error: the
    union raises the depth-1 lower bound, and the measured gap is reported in
    ``validation/validate_bound_bracket.py``. The hand case shows the same
    substitution producing an invalid bound, so the correct form cannot be
    replaced by the union on the strength of this corpus alone.
    """
    from evidencestale.baselines import ground_truth

    items = layered_corpus.evidence
    truth = ground_truth(
        items,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    per_item = changed_sets_by_item(
        layered_corpus.history,
        [(item.identifier, item.produced_at) for item in items],
        layered_corpus.evaluation_time,
    )
    union: set[str] = set()
    for value in per_item.values():
        union |= value
    arguments = {
        "measured": truth.stale_fraction,
        "max_out_degree": layered_corpus.declared_max_dependents,
        "depth_limit": layered_corpus.declared_max_impact_depth,
        "artifact_count": len(layered_corpus.true_graph.nodes),
    }
    correct = stale_fraction_bracket(
        items, per_item, layered_corpus.true_graph.dependencies(), **arguments
    )
    wrong = stale_fraction_bracket(
        items,
        dict.fromkeys(per_item, frozenset(union)),
        layered_corpus.true_graph.dependencies(),
        **arguments,
    )
    assert wrong.lower_depth1 > correct.lower_depth1
    assert wrong.lower_depth0 > correct.lower_depth0
    assert correct.holds is True
