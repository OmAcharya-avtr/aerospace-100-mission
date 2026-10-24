"""The worked example the README quotes, run so the README can quote its output.

The block between the two EXAMPLE markers below is the code in the README, and
the transcript this script writes is the output printed under it. Both are
re-run as the last step before hand-off, so the README cannot drift from the
code: P052 and P056 of this mission each caught their own wrong README numbers
this way.
"""

from __future__ import annotations

import contextlib
import io
import textwrap
from pathlib import Path

import _bootstrap
from _bootstrap import Tee, header

ROOT = _bootstrap.bootstrap()


def main() -> int:
    tee = Tee("worked_example_output.txt")
    header(tee, "cmdguard -- the worked example from README.md, with its real output")

    tee.rule("the example, verbatim")
    source = Path(__file__).read_text().splitlines()
    marker = "# --- EXAMPLE "
    begin = next(i for i, line in enumerate(source) if line.strip() == marker + "BEGIN ---")
    end = next(i for i, line in enumerate(source) if line.strip() == marker + "END ---")
    for line in textwrap.dedent("\n".join(source[begin + 1 : end])).splitlines():
        tee("  " + line)

    tee.rule("its output")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        # --- EXAMPLE BEGIN ---
        from cmdguard import forward_execute, gate_case, load_case, static_lint
        from cmdguard.lint import LintStrength

        case = load_case("examples/cases/late_payload_enable.yaml")
        report = forward_execute(case)

        print(f"steps              : {len(case.sequence)}")
        print(f"violations         : {report.n_violations}")
        print(f"by kind            : {report.kind_counts()}")
        print(f"steps to review    : {report.involved_steps()}")
        print("")

        for violation in report.violations:
            print(violation.render(indent="  "))
        print("")

        # The same case through the per-command lint, which is what people run.
        lint = static_lint(case, LintStrength.WITH_TIMING)
        forward_ids = {(v.step, v.kind.value) for v in report.violations}
        print(f"lint findings      : {lint.n_findings}")
        print(f"confirmed by trace : {len(forward_ids & lint.identities())}")
        print(f"not confirmed      : {len(lint.identities() - forward_ids)}")
        print("")

        # And the gate, which refuses rather than advises.
        print(gate_case(case).render())
        # --- EXAMPLE END ---

    for line in buffer.getvalue().splitlines():
        tee("  " + line)

    tee.rule("the same three calls on the gate-refusal case")
    gate_lines: list[str] = []
    refusal = load_case("examples/cases/irreversible_before_verify.yaml")
    gate_lines.append(gate_case(refusal).render())
    for line in gate_lines:
        for physical in line.splitlines():
            tee("  " + physical)

    tee.rule("checks")
    tee.check("the example case has exactly one violation", report.n_violations == 1)
    tee.check("the trace names the step that caused it",
              report.violations[0].trace[0].step == 5)
    tee.check("the gate does not refuse this case", not gate_case(case).refused)
    tee.check("the gate does refuse the deployment case", gate_case(refusal).refused)
    tee.check("the lint emits more findings than there are violations",
              lint.n_findings > report.n_violations,
              f"{lint.n_findings} findings against {report.n_violations} violation(s)")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
