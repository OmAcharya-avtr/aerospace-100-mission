"""Known answers on hand-constructed cases.

Two cases with answers known by construction, and one worked example whose
answer is the thing the product exists to produce:

* ``planted_gaps.yaml`` was built with exactly one instance of every check.
  The expected code-to-subject mapping is written out below and compared.
* ``complete_minimal.yaml`` was built with every link closed. Expected: zero
  findings, exit 0.
* ``p024_bdot_detumble.yaml`` is the Level 2 worked example. Its eleven
  findings are the genuine gaps in a hazard analysis of P024 DetumbleSim's
  B-dot loop and are NOT filled in. The expected count is pinned so that a
  change to any check is visible here.
"""

from __future__ import annotations

from _support import CASES, FIXTURES, Emitter

from hazardtrace import load_analysis, load_junit, load_requirements, run_checks
from hazardtrace.checks import CHECKS

PLANTED_EXPECTED = {
    "HT001": ("CA-1", "stopped_too_soon_or_applied_too_long"),
    "HT002": ("CA-1", "wrong_timing_or_order"),
    "HT003": ("CA-2", "not_provided"),
    "HT004": ("H-2",),
    "HT005": ("SC-2",),
    "HT006": ("UCA-3",),
    "HT007": ("LS-2",),
    "HT008": ("M-2",),
    "HT009": ("H-3",),
    "HT010": ("UCA-2",),
    "HT011": ("L-2",),
    "HT012": ("REQ-299",),
    "HT013": ("tests/test_planted.py::test_absent",),
    "HT014": ("tests/test_planted.py::test_fails",),
    "HT015": ("CTRL-B",),
    "HT016": ("SNS-ORPHAN",),
}

WORKED_EXPECTED = {
    "HT001": 2, "HT004": 1, "HT005": 1, "HT006": 1, "HT007": 1, "HT008": 1,
    "HT012": 2, "HT013": 1, "HT015": 1,
}


def main() -> int:
    """Run the known-answer checks. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: known answers on hand-constructed cases")
    emit()

    # -- 1. planted case --------------------------------------------------
    planted = run_checks(
        load_analysis(CASES / "planted_gaps.yaml"),
        requirements=load_requirements(FIXTURES / "planted" / "docs" / "REQUIREMENTS.md"),
        tests=load_junit(FIXTURES / "planted" / "junit.xml"),
    )
    emit("A. planted_gaps.yaml: one planted instance of every check")
    emit(f"   checks implemented      : {len(CHECKS)}")
    emit(f"   findings raised         : {len(planted.findings)}")
    emit(f"   exit code               : {planted.exit_code}")
    actual = {f.code: f.subjects for f in planted.findings}
    emit()
    emit(f"   {'code':<8}{'expected subject':<42}{'actual subject':<42}verdict")
    for spec in CHECKS:
        expected = PLANTED_EXPECTED[spec.code]
        got = actual.get(spec.code, ("<absent>",))
        mark = "match" if got == expected else "MISMATCH"
        emit(
            f"   {spec.code:<8}{','.join(expected):<42}{','.join(got):<42}{mark}"
        )
    emit()
    emit.check(
        "every check raises exactly one finding",
        len(planted.findings) == len(CHECKS) == 16,
        f"{len(planted.findings)} findings, {len(CHECKS)} checks",
    )
    emit.check("every planted subject matches", actual == PLANTED_EXPECTED)
    emit.check("planted case exits 1", planted.exit_code == 1)
    counts = planted.count_by_severity()
    emit.check(
        "severity split is 14 error and 2 warning",
        (counts["error"], counts["warning"], counts["info"]) == (14, 2, 0),
        f"{counts['error']} error, {counts['warning']} warning, {counts['info']} info",
    )
    emit.check(
        "enumeration coverage is 7/8",
        planted.grid.completeness() == (7, 8),
        str(planted.coverage.enumeration),
    )
    emit()

    # -- 2. clean case ----------------------------------------------------
    clean = run_checks(load_analysis(CASES / "complete_minimal.yaml"))
    emit("B. complete_minimal.yaml: every link closed")
    emit(f"   findings raised         : {len(clean.findings)}")
    emit(f"   exit code               : {clean.exit_code}")
    emit(f"   enumeration coverage    : {clean.coverage.enumeration}")
    emit.check("clean case raises nothing", clean.findings == ())
    emit.check("clean case exits 0", clean.exit_code == 0)
    emit.check("clean case enumeration is 4/4", clean.grid.completeness() == (4, 4))
    emit()

    # -- 3. worked example ------------------------------------------------
    worked = run_checks(
        load_analysis(CASES / "p024_bdot_detumble.yaml"),
        requirements=load_requirements(FIXTURES / "p024_bdot" / "docs" / "REQUIREMENTS.md"),
        tests=load_junit(FIXTURES / "p024_bdot" / "junit.xml"),
    )
    emit("C. p024_bdot_detumble.yaml: the Level 2 worked example")
    emit("   The findings below are the genuine gaps in this analysis. They are")
    emit("   reported, not fixed; see VALIDATION.md section 6.")
    emit(f"   findings raised         : {len(worked.findings)}")
    emit(f"   exit code               : {worked.exit_code}")
    emit()
    worked_counts = worked.count_by_code()
    for spec in CHECKS:
        if worked_counts[spec.code]:
            emit(f"   {spec.code}  x{worked_counts[spec.code]}  {spec.title}")
    emit()
    for label, ratio in worked.coverage.as_items():
        emit(f"   {label:<32}{ratio}")
    emit()
    emit.check(
        "worked example raises 11 findings",
        len(worked.findings) == 11,
        f"{len(worked.findings)}",
    )
    emit.check(
        "worked example per-code counts are the pinned ones",
        {k: v for k, v in worked_counts.items() if v} == WORKED_EXPECTED,
    )
    emit.check("worked example exits 1", worked.exit_code == 1)
    emit.check(
        "worked example enumeration is 6/8",
        worked.grid.completeness() == (6, 8),
        str(worked.coverage.enumeration),
    )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
