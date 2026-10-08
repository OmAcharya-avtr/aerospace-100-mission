"""Compare the coverage of the three shipped cases and plot it.

Writes ``screenshots/coverage_summary.png``: the four coverage fractions for
each shipped case, with the denominator printed on every bar, and the undefined
fractions drawn as a hatched zero-height marker rather than as a bar at 0 or
100 %, because an undefined fraction is neither.

Run: ``python examples/coverage_summary.py``
"""

from __future__ import annotations

import os

import _bootstrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from assuregraph import load_case, run_checks  # noqa: E402

CASES = ("complete_case.yaml", "incomplete_case.yaml", "cyclic_case.yaml")
METRICS = (
    ("claim_support_fraction", "claims argued\nor undeveloped", "claims_total"),
    ("evidence_present_fraction", "artifacts\npresent", "evidence_total"),
    ("evidence_fresh_fraction", "artifacts fresh\nof checkable", None),
    ("assumption_discharged_fraction", "assumptions\ndischarged", "assumptions_total"),
)


def main() -> int:
    reports = {}
    for filename in CASES:
        case = load_case(os.path.join(_bootstrap.CASES, filename))
        reports[filename] = run_checks(case)

    figure, axes = plt.subplots(figsize=(10.5, 5.2))
    width = 0.26
    colours = ("#2f6b45", "#b3261e", "#8a6d1f")
    for case_index, filename in enumerate(CASES):
        coverage = reports[filename].coverage
        positions = []
        heights = []
        labels = []
        for metric_index, (attribute, _, denominator_attribute) in enumerate(METRICS):
            value = getattr(coverage, attribute)
            if denominator_attribute is None:
                denominator = coverage.evidence_fresh + coverage.evidence_stale
            else:
                denominator = getattr(coverage, denominator_attribute)
            positions.append(metric_index + (case_index - 1) * width)
            heights.append(0.0 if value is None else 100.0 * value)
            labels.append("n/a" if value is None else f"{100.0 * value:.0f}%\nn={denominator}")
        bars = axes.bar(
            positions,
            heights,
            width=width * 0.92,
            color=colours[case_index],
            label=filename,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, text in zip(bars, labels, strict=True):
            axes.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.5,
            )

    axes.set_xticks(range(len(METRICS)))
    axes.set_xticklabels([label for _, label, _ in METRICS], fontsize=9)
    axes.set_ylabel("per cent")
    axes.set_ylim(0, 125)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.legend(loc="upper right", fontsize=8)
    axes.set_title(
        "assuregraph coverage of the three shipped example cases\n"
        "n is the denominator; 'n/a' marks a fraction that is undefined, not zero",
        fontsize=10,
    )
    axes.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    figure.tight_layout()
    target = os.path.join(_bootstrap.SCREENSHOTS, "coverage_summary.png")
    figure.savefig(target, dpi=150)
    plt.close(figure)

    report_lines = [f"wrote {os.path.relpath(target, _bootstrap.REPO_ROOT)}", ""]
    for filename, report in reports.items():
        coverage = report.coverage
        report_lines.append(f"{filename}")
        report_lines.append(
            f"  exit code {report.exit_code}; {report.error_count} error, "
            f"{report.warning_count} warning, {report.info_count} info"
        )
        for attribute, label, denominator_attribute in METRICS:
            value = getattr(coverage, attribute)
            if denominator_attribute is None:
                denominator = coverage.evidence_fresh + coverage.evidence_stale
            else:
                denominator = getattr(coverage, denominator_attribute)
            shown = "undefined" if value is None else f"{100.0 * value:.1f} %"
            flat = " ".join(label.split())
            report_lines.append(f"    {flat:<30} {shown:>10}  denominator {denominator}")
        report_lines.append("")
    text = "\n".join(report_lines)
    with open(
        os.path.join(_bootstrap.VALIDATION, "example_coverage_summary_output.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
