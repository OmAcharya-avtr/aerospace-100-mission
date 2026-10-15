"""Pinned regression tests.

`.hypothesis/` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample Hypothesis found during this build is
pinned here as a named test with its literal inputs written out, along with the
boundary cases that were reasoned about rather than searched for.
"""

from __future__ import annotations

import pytest

from hazardtrace import (
    MIN_JUSTIFICATION_CHARS,
    UCA_CLASSES,
    Analysis,
    CellState,
    Component,
    ComponentKind,
    ControlAction,
    ControlStructure,
    Exclusion,
    FeedbackPath,
    Uca,
    UcaClass,
    build_grid,
    parse_analysis,
    run_checks,
)

CONTROLLER = Component(id="CTRL-1", name="c", kind=ComponentKind.CONTROLLER)
ACTUATOR = Component(id="ACT-1", name="a", kind=ComponentKind.ACTUATOR)
PROCESS = Component(id="PROC-1", name="p", kind=ComponentKind.PROCESS)
ONE_ACTION = ControlStructure(
    components=(CONTROLLER, ACTUATOR, PROCESS),
    control_actions=(
        ControlAction(
            id="CA-0", name="action 0", controller="CTRL-1", actuator="ACT-1",
            process="PROC-1",
        ),
    ),
)


SENSOR = Component(id="SNS-1", name="s", kind=ComponentKind.SENSOR)
ONE_ACTION_CLOSED = ControlStructure(
    components=ONE_ACTION.components + (SENSOR,),
    control_actions=ONE_ACTION.control_actions,
    feedback=(
        FeedbackPath(
            id="FB-1", name="f", sensor="SNS-1", controller="CTRL-1", process="PROC-1"
        ),
    ),
)


def test_filling_an_enumeration_gap_can_increase_the_finding_count():
    """Hypothesis counterexample, 2026-10-15, literal inputs below.

    The claim under test was "adding a UCA never increases the total number of
    findings". It is FALSE, and the smallest falsifying input Hypothesis found
    was one control action with no UCAs and no exclusions:

    * before: 4 x HT001 (all four cells empty) + 1 x HT015 = 5 findings
    * after adding one UCA with no hazard and no loss scenario:
      3 x HT001 + 1 x HT006 + 1 x HT010 + 1 x HT015 = 6 findings

    Filling a cell costs one HT001 and can buy two downstream findings. The
    finding count is therefore **not** a monotone progress measure, and the
    README says so under Limitations. Only the enumeration ratio is monotone.
    """
    before = run_checks(Analysis(title="pinned", structure=ONE_ACTION))
    assert [f.code for f in before.findings] == [
        "HT001", "HT001", "HT001", "HT001", "HT015",
    ]

    after = run_checks(
        Analysis(
            title="pinned",
            structure=ONE_ACTION,
            ucas=(
                Uca(
                    id="U-new", control_action="CA-0", uca_class=UCA_CLASSES[0],
                    description="d", hazards=(),
                ),
            ),
        )
    )
    assert [f.code for f in after.findings] == [
        "HT001", "HT001", "HT001", "HT006", "HT010", "HT015",
    ]
    assert len(after.findings) == len(before.findings) + 1
    # The enumeration ratio, which is the figure the README quotes as progress,
    # does move the right way.
    assert after.coverage.enumeration.numerator == 1
    assert before.coverage.enumeration.numerator == 0


def test_a_conflicting_cell_counts_as_covered_and_is_reported_separately():
    """Pinned decision, not a bug.

    A cell holding both a UCA and an exclusion is CONFLICT. It counts towards
    enumeration coverage, because somebody did consider the cell, and it also
    raises HT003 because the two records disagree. Pinned so that a later change
    to either half is a deliberate change.
    """
    analysis = Analysis(
        title="pinned",
        structure=ONE_ACTION,
        ucas=(
            Uca(
                id="U-0", control_action="CA-0", uca_class=UcaClass.PROVIDED,
                description="d", hazards=(),
            ),
        ),
        exclusions=(
            Exclusion(
                control_action="CA-0", uca_class=UcaClass.PROVIDED,
                justification="a justification long enough to pass the policy",
            ),
        ),
    )
    grid = build_grid(analysis)
    assert grid.count(CellState.CONFLICT) == 1
    assert grid.completeness() == (1, 4)
    codes = [f.code for f in run_checks(analysis).findings]
    assert codes.count("HT003") == 1


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (MIN_JUSTIFICATION_CHARS - 1, ["HT002"]),
        (MIN_JUSTIFICATION_CHARS, []),
        (MIN_JUSTIFICATION_CHARS + 1, []),
    ],
)
def test_justification_length_policy_boundary(length, expected):
    """The policy is >= MIN_JUSTIFICATION_CHARS, pinned at all three boundaries.

    This is an integer comparison on a character count, not a floating-point
    threshold, so the boundary is exact and is asserted as such.
    """
    analysis = Analysis(
        title="pinned",
        structure=ONE_ACTION_CLOSED,
        exclusions=tuple(
            Exclusion(
                control_action="CA-0",
                uca_class=uca_class,
                justification=("x" * length) if index == 0
                else "a justification long enough to pass the policy",
            )
            for index, uca_class in enumerate(UCA_CLASSES)
        ),
    )
    assert [f.code for f in run_checks(analysis).findings] == expected


def test_justification_length_is_measured_after_stripping_whitespace():
    """Pinned: a justification padded with spaces does not pass the policy.

    YAML folded scalars introduce trailing newlines, so the length is taken
    after ``strip()``. Without that, 20 spaces would be an acceptable reason to
    exclude a cell.
    """
    analysis = Analysis(
        title="pinned",
        structure=ONE_ACTION,
        exclusions=(
            Exclusion(
                control_action="CA-0", uca_class=UcaClass.PROVIDED,
                justification="  short  " + " " * 30,
            ),
        ),
    )
    codes = [f.code for f in run_checks(analysis).findings]
    assert codes.count("HT002") == 1


def test_enumeration_ratio_of_seven_eighths_is_compared_with_a_tolerance():
    """Pinned: 7/8 is exact in binary, but the assertion does not rely on that.

    ``Ratio.percent`` is a float produced by a division and a multiplication.
    7/8 happens to be representable, so ``== 87.5`` would pass today; the
    assertion uses a tolerance anyway, because the next ratio added to the
    report will not be representable and the test that copies this one must not
    inherit a bit-exact comparison.
    """
    analysis = parse_analysis(
        "meta:\n  title: t\ncontrol_structure:\n  controllers:\n    - id: C\n"
        "      name: c\n  actuators:\n    - id: A\n      name: a\n"
        "  controlled_processes:\n    - id: P\n      name: p\n  control_actions:\n"
        "    - id: CA-1\n      name: one\n      controller: C\n      actuator: A\n"
        "      process: P\n    - id: CA-2\n      name: two\n      controller: C\n"
        "      actuator: A\n      process: P\n"
        "exclusions:\n"
        + "".join(
            f"  - control_action: {action}\n    uca_class: {uca_class.value}\n"
            "    justification: a justification long enough to pass the policy\n"
            for action, uca_class in (
                ("CA-1", UCA_CLASSES[0]), ("CA-1", UCA_CLASSES[1]),
                ("CA-1", UCA_CLASSES[2]), ("CA-1", UCA_CLASSES[3]),
                ("CA-2", UCA_CLASSES[0]), ("CA-2", UCA_CLASSES[1]),
                ("CA-2", UCA_CLASSES[2]),
            )
        )
    )
    ratio = run_checks(analysis).coverage.enumeration
    assert (ratio.numerator, ratio.denominator) == (7, 8)
    assert ratio.percent == pytest.approx(87.5, abs=1e-9)


def test_an_unreferenced_sensor_adds_exactly_one_warning_and_nothing_else():
    """Pinned from the Hypothesis property of the same name.

    Searched over 100 generated analyses with no falsifying example; the
    literal instance is pinned so the property survives a lost ``.hypothesis``
    directory.
    """
    base = Analysis(title="pinned", structure=ONE_ACTION)
    extended = Analysis(
        title="pinned",
        structure=ControlStructure(
            components=ONE_ACTION.components
            + (Component(id="SNS-ORPHAN", name="orphan", kind=ComponentKind.SENSOR),),
            control_actions=ONE_ACTION.control_actions,
        ),
    )
    before = run_checks(base).findings
    after = run_checks(extended).findings
    added = [f for f in after if f not in before]
    assert [(f.code, f.subjects) for f in added] == [("HT016", ("SNS-ORPHAN",))]
