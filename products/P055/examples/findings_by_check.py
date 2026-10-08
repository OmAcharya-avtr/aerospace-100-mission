"""Plot the findings of the incomplete and cyclic cases, grouped by check and severity.

Writes ``screenshots/findings_by_check.png``. Each bar is one of the six checks;
the stack within it separates error, warning and info, because a declared
Undeveloped goal is an info finding and showing it in the same colour as a
missing artifact would be the kind of chart that gets a case signed off.

Run: ``python examples/findings_by_check.py``
"""

from __future__ import annotations

import os

import _bootstrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from assuregraph import CHECK_NAMES, Severity, load_case, run_checks  # noqa: E402

SEVERITY_COLOURS = {
    Severity.ERROR: "#b3261e",
    Severity.WARNING: "#d79b00",
    Severity.INFO: "#6b7a8f",
}
CASES = ("incomplete_case.yaml", "cyclic_case.yaml")


def main() -> int:
    figure, axes_pair = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    summary_lines = []
    for axes, filename in zip(axes_pair, CASES, strict=True):
        report = run_checks(load_case(os.path.join(_bootstrap.CASES, filename)))
        bottoms = [0] * len(CHECK_NAMES)
        for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
            counts = [
                sum(1 for f in report.by_check(check) if f.severity is severity)
                for check in CHECK_NAMES
            ]
            axes.barh(
                range(len(CHECK_NAMES)),
                counts,
                left=bottoms,
                color=SEVERITY_COLOURS[severity],
                edgecolor="black",
                linewidth=0.4,
                label=severity.value,
            )
            bottoms = [b + c for b, c in zip(bottoms, counts, strict=True)]
        for index, check in enumerate(CHECK_NAMES):
            ids = sorted({i for f in report.by_check(check) for i in f.node_ids})
            if ids:
                axes.annotate(
                    ",".join(ids),
                    (bottoms[index], index),
                    textcoords="offset points",
                    xytext=(5, 0),
                    va="center",
                    fontsize=8,
                )
        axes.set_yticks(range(len(CHECK_NAMES)))
        axes.set_yticklabels(CHECK_NAMES, fontsize=9)
        axes.invert_yaxis()
        axes.set_xlabel("findings")
        axes.set_xlim(0, max(4, max(bottoms) + 2.5))
        axes.set_title(
            f"{filename}\nexit code {report.exit_code}: "
            f"{report.error_count} error, {report.warning_count} warning, "
            f"{report.info_count} info",
            fontsize=10,
        )
        axes.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
        summary_lines.append(f"{filename}: exit {report.exit_code}")
        for check in CHECK_NAMES:
            findings = report.by_check(check)
            if findings:
                for finding in findings:
                    summary_lines.append(
                        f"  {finding.severity.value:<8} {check:<26} "
                        f"{','.join(finding.node_ids)}"
                    )
            else:
                summary_lines.append(f"  {'-':<8} {check:<26} no findings")
        summary_lines.append("")

    handles, labels = axes_pair[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    figure.suptitle(
        "Findings by check, with the offending node ids. "
        "An info finding is a declaration by the case author, not a defect.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    target = os.path.join(_bootstrap.SCREENSHOTS, "findings_by_check.png")
    figure.savefig(target, dpi=150)
    plt.close(figure)

    with open(
        os.path.join(_bootstrap.VALIDATION, "example_findings_by_check_output.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        relative = os.path.relpath(target, _bootstrap.REPO_ROOT)
        handle.write(f"wrote {relative}\n\n" + "\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
