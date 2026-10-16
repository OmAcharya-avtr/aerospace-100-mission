"""Draw the escalation timeline of the worked example against its deadlines.

Writes ``screenshots/escalation_timeline.png``.
"""

from __future__ import annotations

from _bootstrap import SCREENSHOTS, SPECS  # noqa: F401  (sets sys.path)
from watchdogspec import load_spec, render_escalation_png, run_checks


def main() -> int:
    """Render the figure and print what the arithmetic found. Returns 0."""
    spec = load_spec(SPECS / "smallsat_fc.yaml")
    report = run_checks(spec)

    print(f"specification       : {spec.system}")
    print(f"tasks               : {len(spec.tasks)}")
    print(f"ladders             : {len(spec.ladders)}")
    for task_id in sorted(report.timings):
        timing = report.timings[task_id]
        print(
            f"  {task_id:<10} WCDL {timing.detection_latency_ms:8.2f} ms   "
            f"ladder {timing.full_ladder_latency_ms:9.2f} ms   "
            f"D {timing.deadline_ms:8.2f} ms   "
            f"rungs reachable {timing.deepest_rung_reachable}/{len(timing.rungs)}"
        )
    for finding in report.findings:
        print(f"  {finding.code} {finding.severity.value:<7} {finding.subject}")
    print(f"exit code           : {report.exit_code}")

    path = render_escalation_png(spec, report, SCREENSHOTS / "escalation_timeline.png")
    print(f"wrote {path.relative_to(path.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
