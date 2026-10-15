"""Model-level unit tests: the vocabulary, the orderings and the lookups."""

from __future__ import annotations

import pytest

from hazardtrace import (
    UCA_CLASSES,
    Analysis,
    ComponentKind,
    EvidenceKind,
    Loss,
    Uca,
    UcaClass,
)


def test_there_are_exactly_four_uca_classes():
    # The STPA Handbook (Leveson and Thomas, 2018) step 3 defines four classes.
    # The grid arithmetic everywhere in this package multiplies by this number,
    # so it is asserted rather than assumed.
    assert len(UCA_CLASSES) == 4
    assert len(set(UCA_CLASSES)) == 4
    assert set(UCA_CLASSES) == set(UcaClass)


def test_uca_class_order_is_the_handbook_order():
    assert [c.value for c in UCA_CLASSES] == [
        "not_provided",
        "provided",
        "wrong_timing_or_order",
        "stopped_too_soon_or_applied_too_long",
    ]


def test_every_uca_class_has_a_label():
    for uca_class in UCA_CLASSES:
        assert uca_class.label
        assert uca_class.label != uca_class.value or "_" not in uca_class.value


def test_uca_cell_is_action_and_class():
    uca = Uca(id="U", control_action="CA", uca_class=UcaClass.PROVIDED, description="d")
    assert uca.cell == ("CA", UcaClass.PROVIDED)


def test_component_kinds_are_the_four_stpa_roles():
    assert {k.value for k in ComponentKind} == {
        "controller",
        "actuator",
        "sensor",
        "controlled_process",
    }


def test_evidence_kinds():
    assert {k.value for k in EvidenceKind} == {"test", "analysis", "review", "document"}


def test_empty_analysis_lookups_return_none_not_raise():
    analysis = Analysis(title="t")
    assert analysis.uca("nope") is None
    assert analysis.control_action("nope") is None
    assert analysis.component("nope") is None
    assert analysis.cited_requirements() == ()
    assert analysis.cited_tests() == ()


def test_loss_ids_preserve_declaration_order():
    analysis = Analysis(
        title="t",
        losses=(Loss(id="L-9", description="d"), Loss(id="L-1", description="d")),
    )
    assert analysis.loss_ids == ("L-9", "L-1")


def test_model_records_are_frozen():
    loss = Loss(id="L-1", description="d")
    with pytest.raises(AttributeError):
        loss.id = "L-2"
