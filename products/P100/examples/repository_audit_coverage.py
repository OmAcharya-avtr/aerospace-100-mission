"""Plot what the repository audit found and, equally, what it covered.

A finding count without a denominator is the thing this product exists to
object to, so the figure gives both: how many candidate claims were evaluated,
how many could not be, and where the findings sit against that denominator.

Reads ``validation/repository_findings.json``, which
``validation/validate_repository_audit.py`` writes. Nothing here re-runs the
audit; this example draws the committed evidence.

Writes ``screenshots/repository_coverage.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
_bootstrap.style()

import matplotlib.pyplot as plt  # noqa: E402

from seedaudit.power import escaped_rate_upper_bound  # noqa: E402

#: The evidence file, relative to the product root.
EVIDENCE = "validation/repository_findings.json"

#: Colours for the coverage statuses.
STATUS_COLOURS = {
    "compared": "#92c5de",
    "no_output": "#f7c9a8",
    "run_failed": "#ef8a62",
    "timeout": "#b2182b",
}


def main() -> int:
    """Draw the audit's findings and its coverage side by side."""
    path = ROOT / EVIDENCE
    if not path.is_file():
        print(f"{_bootstrap.relative(path)} is absent.")
        print("Run validation/validate_repository_audit.py first; it writes the evidence.")
        return 1
    evidence = json.loads(path.read_text(encoding="utf-8"))

    counts = evidence["status_counts"]
    found = evidence["candidates_found"]
    attempted = evidence["candidates_attempted"]
    print(f"repository: {evidence['root_name']}")
    print(f"roots scanned: {', '.join(evidence['roots'])}")
    for item in evidence["exclusions"]:
        print(f"excluded: {item}")
    print(f"candidates found: {found}")
    print(f"candidates attempted: {attempted}")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    print(f"seed-attributable moved numbers: {evidence['headline_seed_attributable']}")
    print(f"declared-seed moved numbers: {evidence['headline_declared_seed']}")
    print(f"claim tokens across all candidates: {evidence['claim_tokens_total']}")
    print(f"claim tokens evaluated: {evidence['claim_tokens_compared']}")
    print(f"static findings: {evidence['static_findings_total']}")
    print(f"  in {evidence['static_files_with_findings']} files")
    print(f"  of {evidence['static_files_parsed']} parsed")
    print(f"escaped-rate upper bound on the repeat axis: {evidence['escaped_bound']!r}")

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), width_ratios=[1.0, 1.0, 1.1])

    left = axes[0]
    statuses = sorted(counts, key=lambda key: -counts[key])
    values = [counts[key] for key in statuses]
    unattempted = found - attempted
    if unattempted > 0:
        statuses.append("not attempted")
        values.append(unattempted)
        STATUS_COLOURS["not attempted"] = "0.75"
    bottom = 0.0
    for status, value in zip(statuses, values, strict=True):
        left.bar(
            0,
            value,
            0.55,
            bottom=bottom,
            color=STATUS_COLOURS.get(status, "0.6"),
            label=f"{status} ({value})",
        )
        bottom += value
    left.set_xticks([])
    left.set_ylabel("candidate claims")
    left.set_title(f"Coverage: {found} candidates found")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncols=2, fontsize=7.5)
    left.grid(False)

    middle = axes[1]
    labels = ["moved under\na seed change", "moved at the\ndeclared seed"]
    found_counts = [
        evidence["headline_seed_attributable"],
        evidence["headline_declared_seed"],
    ]
    middle.bar(labels, found_counts, 0.5, color=["#b2182b", "#ef8a62"])
    for index, value in enumerate(found_counts):
        middle.annotate(
            str(value),
            xy=(index, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    middle.set_ylabel("committed numbers")
    middle.set_title(
        f"Findings against {evidence['claim_tokens_compared']} evaluated claims"
    )
    middle.set_ylim(0, max(max(found_counts), 1) * 1.25)
    middle.grid(False)

    right = axes[2]
    repeats = np.arange(1, 201)
    right.plot(
        repeats,
        [escaped_rate_upper_bound(int(n)) for n in repeats],
        lw=1.8,
        color="#b2182b",
    )
    achieved = max(1, evidence["repeats_per_claim"])
    right.scatter([achieved], [evidence["escaped_bound"]], s=36, color="#2166ac", zorder=5)
    right.annotate(
        f"this audit: n = {achieved}\nbound {evidence['escaped_bound']:.4f}",
        xy=(achieved, evidence["escaped_bound"]),
        xytext=(16, -34),
        textcoords="offset points",
        fontsize=8,
        color="#2166ac",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("agreeing repeats n")
    right.set_ylabel("upper bound on the escaped rate")
    right.set_title("What a clean verdict here has earned")

    figure.tight_layout()
    target = ROOT / _bootstrap.SCREENSHOTS / "repository_coverage.png"
    figure.savefig(target)
    plt.close(figure)
    print()
    print(f"figure written to {_bootstrap.relative(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
