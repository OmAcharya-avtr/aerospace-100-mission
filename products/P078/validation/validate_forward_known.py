"""Forward execution against hand-constructed sequences with known violations.

Every expectation in this script is derived by hand from the committed case
files, and the derivation is written out next to the assertion so a reader can
check the arithmetic without running anything. The three cases cover, between
them, eight of the ten violation kinds; the remaining two,
``undeclared_command`` and ``illegal_transition``, are covered by the inline
cases at the end.
"""

from __future__ import annotations

import _bootstrap
from _bootstrap import Tee, header, relative

ROOT = _bootstrap.bootstrap()

from cmdguard.lint import LintStrength, static_lint  # noqa: E402
from cmdguard.loader import load_case  # noqa: E402
from cmdguard.model import (  # noqa: E402
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
)
from cmdguard.report import render_check  # noqa: E402
from cmdguard.trace import ViolationKind, forward_execute, peak_draw  # noqa: E402

CASES = ROOT / "examples" / "cases"


def main() -> int:
    tee = Tee("validate_forward_known_output.txt")
    header(tee, "cmdguard -- forward execution against hand-constructed cases")

    # ------------------------------------------------------------------
    tee.rule("1. nominal_science_pass -- the negative control")
    case = load_case(CASES / "nominal_science_pass.yaml")
    report = forward_execute(case)
    tee(f"  case file : {relative(CASES / 'nominal_science_pass.yaml')}")
    tee(f"  steps     : {len(case.sequence)}, span {case.sequence.duration_s:.1f} s")
    tee(f"  violations: {report.n_violations}")
    tee.check("negative control reports no violation", report.n_violations == 0,
              f"kinds={report.kind_counts()}")
    peaks = peak_draw(report)
    tee("")
    tee("  hand derivation of the peak held power, at t=80 s:")
    tee("    THERM_HEATER_ON  t=10 s, hold 300 s, draw 24.0 W -> live")
    tee("    ADCS_ATT_VERIFY  t=20 s, hold  20 s, draw  8.0 W -> expired at t=40 s")
    tee("    PAYLOAD_ENABLE   t=40 s, hold  90 s, draw 18.0 W -> live")
    tee("    PAYLOAD_SCAN     t=80 s,             draw 46.0 W -> this step")
    tee("    total = 24.0 + 18.0 + 46.0 = 88.0 W against a declared 120.0 W")
    tee(f"  measured peak power_w        : {peaks['power_w']:.6f}")
    tee(f"  measured peak data_rate_mbps : {peaks['data_rate_mbps']:.6f}")
    tee.check("peak power equals the hand derivation", abs(peaks["power_w"] - 88.0) < 1e-9)
    tee.check("peak data rate equals the declared scan draw",
              abs(peaks["data_rate_mbps"] - 22.0) < 1e-9)
    tee.check("final mode is NOMINAL", report.final_mode == "NOMINAL", report.final_mode)

    # ------------------------------------------------------------------
    tee.rule("2. late_payload_enable -- one ordering defect, and its trace")
    case = load_case(CASES / "late_payload_enable.yaml")
    report = forward_execute(case)
    tee(f"  violations: {report.n_violations}  kinds: {report.kind_counts()}")
    tee.check("exactly one violation", report.n_violations == 1)
    violation = report.violations[0]
    tee.check("the violating step is step 6", violation.step == 6, f"step {violation.step}")
    tee.check("the kind is precondition_false",
              violation.kind.value == "precondition_false", violation.kind.value)
    tee.check("the command is PAYLOAD_SCAN", violation.command == "PAYLOAD_SCAN")
    first = violation.trace[0]
    tee.check("the trace names step 5 as the cause", first.step == 5, f"step {first.step}")
    tee.check("the trace names PAYLOAD_SAFE", first.command == "PAYLOAD_SAFE")
    tee.check("the trace names t=80.0 s", abs((first.t_s or -1.0) - 80.0) < 1e-9)
    tee("")
    tee("  the rendered finding, which is the deliverable:")
    for line in violation.render(indent="      ").splitlines():
        tee(f"    {line}")
    tee("")
    lint = static_lint(case, LintStrength.PER_COMMAND)
    forward_ids = {(v.step, v.kind.value) for v in report.violations}
    lint_ids = lint.identities()
    tee(f"  per-command lint findings on the same case: {len(lint_ids)}")
    for finding in lint.findings:
        tee(f"    {finding.render()}")
    tee(f"  findings the forward walk confirms   : {len(forward_ids & lint_ids)}")
    tee(f"  findings the forward walk contradicts: {len(lint_ids - forward_ids)}")
    tee.check("the lint reports the legal PAYLOAD_ENABLE as a violation",
              (3, "precondition_false") in lint_ids,
              "a false positive: heater_on is false at t=0 and true by t=40 s")

    # ------------------------------------------------------------------
    tee.rule("3. resource_and_exclusion -- six hand-derived violations")
    case = load_case(CASES / "resource_and_exclusion.yaml")
    report = forward_execute(case)
    tee(f"  violations: {report.n_violations}  kinds: {report.kind_counts()}")
    by_key = {(v.step, v.kind.value): v for v in report.violations}
    expectations = [
        (
            (2, "mutual_exclusion"),
            150.0,
            "HEATER_A took thermal_pair at t=10 s for 200 s, so it is held until "
            "t=210 s; HEATER_B at t=60 s is 150 s early",
        ),
        (
            (3, "resource_limit"),
            35.0,
            "HEATER_A 40 W + HEATER_B 40 W + XMIT 55 W = 135 W against a 100 W "
            "ceiling, over by 35 W",
        ),
        (
            (4, "timing_gap"),
            10.0,
            "XMIT declares min_gap_after_s = 20 s and ran at t=70 s; SSR_WRITE at "
            "t=80 s is 10 s short",
        ),
        (
            (5, "resource_limit"),
            15.0,
            "SSR_WRITE 30 Mbps (held 80-140 s) + SSR_READ 25 Mbps = 55 Mbps against "
            "40 Mbps, over by 15 Mbps",
        ),
        (
            (6, "timing_interval"),
            170.0,
            "XMIT declares min_interval_s = 300 s and last ran at t=70 s; the repeat "
            "at t=200 s is 170 s early",
        ),
        (
            (6, "resource_limit"),
            35.0,
            "both heaters are still live at t=200 s (until 210 s and 260 s), so the "
            "repeat crosses the ceiling again",
        ),
    ]
    tee.check("exactly six violations", report.n_violations == 6, str(report.n_violations))
    for key, shortfall, derivation in expectations:
        found = by_key.get(key)
        tee("")
        tee(f"  expected {key[1]} at step {key[0]}, shortfall {shortfall:.6f}")
        tee(f"    hand derivation: {derivation}")
        if found is None:
            tee.check(f"{key[1]} at step {key[0]} is reported", False, "not reported")
            continue
        tee.check(f"{key[1]} at step {key[0]} is reported", True)
        tee.check(
            f"{key[1]} at step {key[0]} shortfall",
            found.shortfall is not None and abs(found.shortfall - shortfall) < 1e-9,
            f"measured {found.shortfall:.6f}",
        )
    tee("")
    tee.check(
        "no command in this dictionary exceeds a ceiling on its own",
        all(
            float(spec.resource_draw.get(key, 0.0))
            <= (case.dictionary.resource_limits.limit_for(key) or float("inf"))
            for spec in case.dictionary.commands.values()
            for key in spec.draw_keys
        ),
        "so the lint's single-command resource check cannot fire here",
    )
    lint = static_lint(case, LintStrength.WITH_TIMING)
    kinds = set(lint.kind_counts())
    tee.check("the lint finds no mutual exclusion here", "mutual_exclusion" not in kinds)
    tee.check("the lint finds no accumulated overdraw here", "resource_limit" not in kinds)
    tee.check("the lint with timing does find both timing defects",
              kinds == {"timing_gap", "timing_interval"}, str(sorted(kinds)))

    # ------------------------------------------------------------------
    tee.rule("4. the two remaining kinds, on inline cases")
    machine = ModeMachine(modes=("SAFE", "NOMINAL", "SCIENCE"),
                          transitions={"SAFE": ("NOMINAL",), "NOMINAL": ("SAFE",), "SCIENCE": ()})
    jump = CommandSpec(identifier="JUMP_SCIENCE", mode_effect="SCIENCE")
    inline = Case(
        name="inline_illegal_transition",
        dictionary=CommandDictionary(
            commands={"JUMP_SCIENCE": jump}, resource_limits=ResourceLimits({})
        ),
        machine=machine,
        initial=InitialState(mode="NOMINAL", predicates={}),
        sequence=CommandSequence(
            (
                SequenceStep(t_s=0.0, command="JUMP_SCIENCE"),
                SequenceStep(t_s=10.0, command="TYPO_COMMAND"),
            )
        ),
    )
    report = forward_execute(inline)
    tee(f"  violations: {report.n_violations}  kinds: {report.kind_counts()}")
    tee.check("illegal_transition is reported",
              report.kind_counts().get("illegal_transition") == 1)
    tee.check("undeclared_command is reported",
              report.kind_counts().get("undeclared_command") == 1)
    tee.check("the forbidden transition is not applied", report.final_mode == "NOMINAL",
              report.final_mode)
    tee("")
    tee("  every violation kind this package reports has now been exercised:")
    exercised = {"precondition_false", "precondition_unknown", "irreversible_unverified",
                 "mode_gate", "illegal_transition", "timing_gap", "timing_interval",
                 "mutual_exclusion", "resource_limit", "undeclared_command"}
    declared = {kind.value for kind in ViolationKind}
    tee(f"    declared : {len(declared)}")
    tee(f"    exercised: {len(exercised & declared)} (here or in the gate script)")
    tee.check("every declared kind is exercised somewhere in validation/",
              exercised == declared, str(sorted(declared - exercised)))

    # ------------------------------------------------------------------
    tee.rule("5. a rendered report, verbatim")
    case = load_case(CASES / "late_payload_enable.yaml")
    for line in render_check(forward_execute(case), case).splitlines():
        tee(f"  {line}")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
