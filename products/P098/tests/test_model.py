"""Input validation of the data types, and the GSN restrictions they enforce."""

from __future__ import annotations

import pytest

from evidencestale.model import (
    LINK_TYPES,
    NODE_TYPES,
    AssuranceCase,
    ChangeEvent,
    EvidenceItem,
)


def test_write_event_requires_content() -> None:
    with pytest.raises(ValueError, match="must carry content"):
        ChangeEvent(time=1, artifact="a", kind="write")


def test_touch_event_forbids_content() -> None:
    with pytest.raises(ValueError, match="must not carry content"):
        ChangeEvent(time=1, artifact="a", kind="touch", content="x")


def test_unknown_event_kind_names_the_allowed_kinds() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ChangeEvent(time=1, artifact="a", kind="delete", content="x")  # type: ignore[arg-type]


def test_negative_revision_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative revision"):
        ChangeEvent(time=-1, artifact="a", kind="touch")


def test_revision_must_be_int_not_float() -> None:
    with pytest.raises(TypeError, match="int revision index"):
        ChangeEvent(time=1.0, artifact="a", kind="touch")  # type: ignore[arg-type]


def test_bool_is_not_an_acceptable_revision() -> None:
    # bool is a subclass of int in Python; a revision of True would silently
    # become revision 1, so it is rejected explicitly.
    with pytest.raises(TypeError, match="int revision index"):
        ChangeEvent(time=True, artifact="a", kind="touch")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", " a", "a "])
def test_identifier_must_be_non_empty_and_untrimmed(bad: str) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ChangeEvent(time=1, artifact=bad, kind="touch")


def test_evidence_item_rejects_negative_production_revision() -> None:
    with pytest.raises(ValueError, match="non-negative revision"):
        EvidenceItem(
            identifier="e", subject="a", produced_at=-3, goal="G", solution="Sn"
        )


def test_evidence_item_true_stale_must_be_bool_or_none() -> None:
    with pytest.raises(TypeError, match="bool or None"):
        EvidenceItem(
            identifier="e",
            subject="a",
            produced_at=0,
            goal="G",
            solution="Sn",
            true_stale=1,  # type: ignore[arg-type]
        )


def test_node_types_and_link_types_are_the_modelled_subset() -> None:
    # Fixed by src/evidencestale/model.py against the GSN Community Standard
    # v3 node and link vocabulary. A change here is a change of scope and must
    # be reflected in README.md under Limitations.
    assert NODE_TYPES == (
        "Goal",
        "Strategy",
        "Solution",
        "Context",
        "Assumption",
        "Justification",
    )
    assert LINK_TYPES == ("SupportedBy", "InContextOf")


def test_unknown_node_type_rejected() -> None:
    with pytest.raises(ValueError, match="not one of the"):
        AssuranceCase(nodes={"M1": "Module"}, links=())


def test_link_to_unknown_node_rejected() -> None:
    with pytest.raises(ValueError, match="unknown node as its target"):
        AssuranceCase(nodes={"G1": "Goal"}, links=(("SupportedBy", "G1", "Sn1"),))


def test_solution_may_not_support_another_node() -> None:
    with pytest.raises(ValueError, match="originates at a"):
        AssuranceCase(
            nodes={"Sn1": "Solution", "Sn2": "Solution"},
            links=(("SupportedBy", "Sn1", "Sn2"),),
        )


def test_in_context_of_may_not_terminate_at_a_solution() -> None:
    with pytest.raises(ValueError, match="terminates at a"):
        AssuranceCase(
            nodes={"G1": "Goal", "Sn1": "Solution"},
            links=(("InContextOf", "G1", "Sn1"),),
        )


def test_malformed_link_triple_rejected() -> None:
    with pytest.raises(ValueError, match="triple"):
        AssuranceCase(nodes={"G1": "Goal"}, links=(("SupportedBy", "G1"),))  # type: ignore[arg-type]


def test_leaf_goals_are_the_goals_whose_children_are_all_solutions() -> None:
    case = AssuranceCase(
        nodes={
            "G0": "Goal",
            "G1": "Goal",
            "G2": "Goal",
            "Sn1": "Solution",
            "Sn2": "Solution",
        },
        links=(
            ("SupportedBy", "G0", "G1"),
            ("SupportedBy", "G0", "G2"),
            ("SupportedBy", "G1", "Sn1"),
            ("SupportedBy", "G2", "Sn2"),
        ),
    )
    # G0's children are goals, so it is not a leaf goal; G1 and G2 rest
    # directly on evidence, so they are.
    assert case.leaf_goals() == ("G1", "G2")
    assert case.solutions() == ("Sn1", "Sn2")
