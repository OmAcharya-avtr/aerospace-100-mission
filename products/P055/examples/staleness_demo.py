"""Walk one artifact through every evidence state and plot the timeline.

Writes ``screenshots/staleness_timeline.png``. The point of the picture is the
pair of steps in the middle: a change that inverts the artifact's verdict and a
change that only adds a comment line produce the same ``stale`` state. The
content hash is a change detector, and the chart is drawn so that nobody can
read it as a materiality judgement.

Run: ``python examples/staleness_demo.py``
"""

from __future__ import annotations

import os
import shutil
import tempfile

import _bootstrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from assuregraph import (  # noqa: E402
    EvidenceStatus,
    inspect_evidence,
    parse_case,
    sha256_file,
)

STATE_ORDER = [
    EvidenceStatus.FRESH,
    EvidenceStatus.STALE,
    EvidenceStatus.UNVERIFIABLE,
    EvidenceStatus.ABSENT,
]
STATE_COLOUR = {
    EvidenceStatus.FRESH: "#2f6b45",
    EvidenceStatus.STALE: "#b3261e",
    EvidenceStatus.UNVERIFIABLE: "#d79b00",
    EvidenceStatus.ABSENT: "#555555",
}

ORIGINAL = "monitor acceptance report\nverdict: pass\nviolations: 0\n"


def _case(base_dir: str, digest: str | None):
    evidence: dict[str, object] = {"path": "report.txt"}
    if digest is not None:
        evidence["sha256"] = digest
    return parse_case(
        {
            "name": "staleness walk",
            "nodes": [
                {
                    "id": "Sn1",
                    "type": "solution",
                    "statement": "monitor acceptance report",
                    "evidence": evidence,
                }
            ],
        },
        base_dir=base_dir,
    )


def main() -> int:
    steps: list[tuple[str, EvidenceStatus, str]] = []
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "report.txt")

        def write(content: str) -> None:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)

        def status(digest: str | None) -> tuple[EvidenceStatus, str]:
            case = _case(scratch, digest)
            report = inspect_evidence(case, case.nodes["Sn1"])
            return report.status, (report.actual_sha256 or "")[:10]

        write(ORIGINAL)
        recorded = sha256_file(path)
        state, found = status(recorded)
        steps.append(("1. cited, unchanged", state, found))

        write(ORIGINAL + "# regenerated 2026-10-08\n")
        state, found = status(recorded)
        steps.append(("2. comment line added\n(meaning unchanged)", state, found))

        write(ORIGINAL)
        state, found = status(recorded)
        steps.append(("3. comment removed\n(back to the cited bytes)", state, found))

        write(ORIGINAL.replace("verdict: pass", "verdict: fail"))
        state, found = status(recorded)
        steps.append(("4. verdict inverted\n(meaning reversed)", state, found))

        write(ORIGINAL)
        state, found = status(None)
        steps.append(("5. no digest recorded\nin the case", state, found))

        shutil.move(path, path + ".bak")
        state, found = status(recorded)
        steps.append(("6. artifact deleted", state, found))

    figure, axes = plt.subplots(figsize=(11.0, 4.4))
    positions = range(len(steps))
    heights = [STATE_ORDER.index(state) for _, state, _ in steps]
    axes.step(positions, heights, where="mid", color="#444444", linewidth=1.0, zorder=1)
    for index, (_label, state, digest) in enumerate(steps):
        axes.scatter(
            [index],
            [STATE_ORDER.index(state)],
            s=240,
            color=STATE_COLOUR[state],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        axes.annotate(
            state.value + (f"\n{digest}…" if digest else ""),
            (index, STATE_ORDER.index(state)),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=8,
        )
    axes.set_xticks(list(positions))
    axes.set_xticklabels([label for label, _, _ in steps], fontsize=8.5)
    axes.set_yticks(range(len(STATE_ORDER)))
    axes.set_yticklabels([state.value for state in STATE_ORDER], fontsize=9)
    axes.set_ylim(-0.6, len(STATE_ORDER) - 0.1)
    axes.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axes.set_title(
        "One artifact through every evidence state.\n"
        "Steps 2 and 4 are the same finding: the hash detects that the bytes moved, "
        "not whether the move mattered.",
        fontsize=10,
    )
    figure.tight_layout()
    target = os.path.join(_bootstrap.SCREENSHOTS, "staleness_timeline.png")
    figure.savefig(target, dpi=150)
    plt.close(figure)

    lines = [
        f"wrote {os.path.relpath(target, _bootstrap.REPO_ROOT)}",
        "",
        f"{'step':<44} {'state':<14} current digest (first 10)",
    ]
    lines.append("-" * 82)
    for label, state, digest in steps:
        lines.append(f"{' '.join(label.split()):<44} {state.value:<14} {digest or 'none'}")
    lines.append("")
    lines.append(
        "Steps 2 and 4 are both 'stale'. The content hash reports that the bytes "
        "changed; it does not and cannot report whether the change matters."
    )
    lines.append(
        "Step 3 returns to 'fresh': the check reads current bytes only and holds no "
        "history, so a change that was reverted leaves no trace."
    )
    with open(
        os.path.join(_bootstrap.VALIDATION, "example_staleness_demo_output.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
