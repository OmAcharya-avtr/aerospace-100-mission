"""Level 2 worked example: the B-dot detumble loop of P024 DetumbleSim.

Carried end to end - control structure, four-class UCA enumeration, loss
scenarios, mitigations, evidence - and reported with the gaps it actually has.
Nothing below was tidied to make the example look finished. The eleven findings
are the output; eight of them correspond to functions P024 does not have.

The block between the EXAMPLE markers is the code quoted in README.md under
"A worked example". Its printed output is the text committed beside this script.
"""

from __future__ import annotations

from _support import CASES, FIXTURES, Emitter

from hazardtrace import (
    format_coverage_table,
    format_grid,
    load_analysis,
    load_junit,
    load_requirements,
    run_checks,
)

ANALYSIS = CASES / "p024_bdot_detumble.yaml"
REQUIREMENTS = FIXTURES / "p024_bdot" / "docs" / "REQUIREMENTS.md"
JUNIT = FIXTURES / "p024_bdot" / "junit.xml"

# The eight gaps that correspond to something P024 does not implement, each
# with the check that found it and the source fact behind it. Written out so
# that the narrative and the computed findings can be compared line by line.
GAP_NOTES: tuple[tuple[str, str, str], ...] = (
    (
        "HT001",
        "CA-DIPOLE x stopped too soon / applied too long",
        "P024 ends actuation when simulate.py reaches duration_s. There is no "
        "onboard termination of the dipole command, no timeout and no declared "
        "response when the rate threshold is never reached, so there is nothing "
        "in the source to write a UCA context against and nothing that would "
        "justify excluding the class.",
    ),
    (
        "HT001",
        "CA-EXIT x wrong timing or order",
        "P024 has no mode sequencing at all: detumble entry and exit are not "
        "implemented, and detumble_time_s is computed in post-processing from "
        "the true body rate. There is no order of operations to analyse.",
    ),
    (
        "HT004",
        "H-4 has no system-level constraint",
        "H-4 is the body rate increasing while the controller is active. P024's "
        "own test suite carries an explicit counterexample in which |J omega| "
        "rises for an asymmetric inertia under the ideal law, and there is no "
        "divergence monitor anywhere in the package, so no constraint can be "
        "traced to anything.",
    ),
    (
        "HT005",
        "SC-5 cites no requirement identifier",
        "SC-5 says the magnetic measurement must not be corrupted by the dipole "
        "the control law is commanding. P024 models no coil-to-magnetometer "
        "cross-talk, so there is no requirement to cite and the constraint is "
        "an unbacked assertion.",
    ),
    (
        "HT006",
        "UCA-3 has no loss scenario",
        "UCA-3 is the command being dominated by magnetometer noise late in the "
        "detumble. The backward difference divides per-sample noise by the "
        "control interval, so the noise term in dB/dt scales as 1/dt, but P024 "
        "declares no noise floor, no filter and no deadband, so the causal "
        "scenario cannot be written from the source.",
    ),
    (
        "HT007",
        "LS-5 has no mitigation",
        "LS-5 is the magnetometer sampling the field while a coil is driven. "
        "P024 neither models nor mitigates this; there is no sample-and-settle "
        "schedule in simulate.py, which samples and commands at the same instant.",
    ),
    (
        "HT008",
        "M-4 cites no evidence artifact",
        "M-4 bounds the deployment tumble rate by operational agreement. That is "
        "a procedural control with nothing in the repository behind it, which is "
        "exactly the case this check exists to surface.",
    ),
    (
        "HT015",
        "CTRL-MODE has no feedback path",
        "The mode manager is declared open loop, and it genuinely is: B-dot is a "
        "magnetometer-only law, P024 carries no rate estimator, and the exit "
        "criterion is evaluated on the true state. The controller that decides "
        "detumble is finished has no measurement of the quantity it decides on.",
    ),
)

CROSS_LINK_NOTES: tuple[tuple[str, str, str], ...] = (
    (
        "HT012",
        "REQ-021 is not declared",
        "Cited by M-5, the proposed divergence monitor. The requirement does not "
        "exist because the function does not exist.",
    ),
    (
        "HT012",
        "REQ-022 is not declared",
        "Cited by SC-4, the energy-exhaustion constraint. P024 has no power or "
        "energy model at all.",
    ),
    (
        "HT013",
        "tests/test_divergence_monitor.py::test_rate_increase_commands_safe_mode",
        "Cited by EV-6. Absent from a junit report of 305 real P024 tests, "
        "because the test and the code it would test are both unwritten.",
    ),
)


def main() -> int:
    """Run the worked example. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace Level 2 worked example")
    emit("STPA of the B-dot detumble loop implemented in P024 DetumbleSim")
    emit()
    emit("Method: STPA Handbook, Leveson and Thomas, 2018 (four steps, four UCA")
    emit("classes). Accident model: STAMP, Leveson, Engineering a Safer World,")
    emit("MIT Press, 2011.")
    emit()
    emit("This analysis is INCOMPLETE and is published incomplete. The findings")
    emit("below are its real gaps. Research-grade: nothing here is flight")
    emit("qualified, certified or approved for operational aerospace use, and")
    emit("none of it discharges any safety obligation.")
    emit()

    # ---- EXAMPLE BEGIN (quoted in README.md) ----
    analysis = load_analysis("examples/cases/p024_bdot_detumble.yaml")
    requirements = load_requirements("fixtures/p024_bdot/docs/REQUIREMENTS.md")
    tests = load_junit("fixtures/p024_bdot/junit.xml")
    report = run_checks(analysis, requirements=requirements, tests=tests)

    emit(f"control actions   : {len(analysis.structure.control_actions)}")
    emit(f"feedback paths     : {len(analysis.structure.feedback)}")
    emit(f"UCAs / exclusions  : {len(analysis.ucas)} / {len(analysis.exclusions)}")
    emit(f"enumeration        : {report.coverage.enumeration}")
    emit(f"complete           : {report.is_complete}   exit code: {report.exit_code}")
    emit()
    emit(format_grid(report).rstrip("\n"))
    emit()
    emit(format_coverage_table(report.coverage).rstrip("\n"))
    emit()
    for finding in report.findings:
        emit(f"  {finding.code}  {finding.severity.value:<7} {finding.message}")
    # ---- EXAMPLE END ----

    emit()
    emit("=" * 78)
    emit("The gaps, stated plainly, with the source fact behind each")
    emit("=" * 78)
    emit()
    emit("A. gaps inside the analysis (8 findings)")
    emit()
    for code, subject, note in GAP_NOTES:
        emit(f"   {code}  {subject}")
        for line in _wrap(note, 70):
            emit(f"         {line}")
        emit()
    emit("B. gaps found by cross-linking against the repository (3 findings)")
    emit()
    for code, subject, note in CROSS_LINK_NOTES:
        emit(f"   {code}  {subject}")
        for line in _wrap(note, 70):
            emit(f"         {line}")
        emit()

    emit("C. what the analysis does NOT claim")
    emit()
    for line in _wrap(
        "Every UCA and loss scenario above was written from P024's source and "
        "docstrings. None of them was reviewed by anybody, none is traced to a "
        "hazard log, and the losses and hazards are the author's judgement about "
        "a 3U-class mission that does not exist. The analysis is a demonstration "
        "that the checks compute something real on a real control loop. It is "
        "not a hazard analysis of any flight system, and no part of it should be "
        "presented as one.",
        74,
    ):
        emit(f"   {line}")
    emit()

    emit("D. verification of this script's own claims")
    emit()
    codes = report.count_by_code()
    emit.check(
        "the analysis parses and every internal reference resolves",
        analysis.title.startswith("B-dot"),
    )
    emit.check(
        "11 findings are raised",
        len(report.findings) == 11,
        f"{len(report.findings)}",
    )
    emit.check(
        "the narrative in A names every in-analysis finding",
        sorted(code for code, _s, _n in GAP_NOTES)
        == sorted(
            f.code for f in report.findings if f.code not in {"HT012", "HT013", "HT014"}
        ),
    )
    emit.check(
        "the narrative in B names every cross-link finding",
        sorted(code for code, _s, _n in CROSS_LINK_NOTES)
        == sorted(
            f.code for f in report.findings if f.code in {"HT012", "HT013", "HT014"}
        ),
    )
    emit.check(
        "enumeration coverage is 6 of 8 cells",
        report.grid.completeness() == (6, 8),
        str(report.coverage.enumeration),
    )
    emit.check(
        "two enumeration cells are unanalysed",
        codes["HT001"] == 2,
        f"{codes['HT001']}",
    )
    emit.check("the exit code is 1", report.exit_code == 1)
    emit.check(
        "the junit report cross-linked against holds 305 cases",
        len(tests.cases) == 305,
        f"{len(tests.cases)}",
    )
    return emit.finish()


def _wrap(text: str, width: int) -> list[str]:
    """Wrap ``text`` to ``width`` characters without importing textwrap."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
