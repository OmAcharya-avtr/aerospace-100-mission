"""Measure the cost of a run against case size, so the compute budget is on record.

Synthetic cases of increasing size are generated, written to disk with real
evidence artifacts, and checked end to end. The quantities recorded are wall
clock seconds on two shared contended cores, which move by 10-20 % run to run;
the committed output is the run the README and VALIDATION.md tables were taken
from. The complexity claims the modules document (O(V+E) for Tarjan and for
breadth-first search, one artifact read per Solution per run) are what this
table is evidence for, within the size range tested and no further.

Validity range: about 150 to about 30000 nodes, with one 1 KiB evidence
artifact per claim leaf. Nothing here is a measurement of a real assurance case;
published cases run to tens or low hundreds of elements, so the largest size
tested is a stress case rather than a representative one.

Run: ``python validation/validate_scale.py``
Output: ``validation/validate_scale_output.txt``
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time

import _bootstrap  # noqa: F401
import yaml

from assuregraph import LOADER_NAME, load_case, render_mermaid, run_checks

SIZES = (150, 750, 3000, 7500, 30000)
ARTIFACT_BYTES = 1024


def build_case(directory: str, node_count: int) -> tuple[str, int, int]:
    """Write a synthetic case of about ``node_count`` nodes.

    Structure: the claims form a complete binary tree under ``SupportedBy``,
    alternating Goal and Strategy; every claim that is a leaf of that tree gets
    exactly one Solution child whose artifact is written to disk with its
    matching digest. The result therefore has no unsupported claim, no missing
    or stale artifact, no undischarged assumption, no cycle and no orphan, so a
    correct run reports zero findings at any size. A nonzero finding count means
    the generator or a check is wrong, and the script fails.

    Returns:
        ``(path, total nodes, solution count)``.
    """
    evidence_dir = os.path.join(directory, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    # claims + leaves == node_count, with leaves == ceil(claims / 2).
    claims = max(1, round(node_count / 1.5))
    nodes: list[dict[str, object]] = [
        {
            "id": f"N{i}",
            "type": "goal" if i % 2 == 0 else "strategy",
            "statement": f"synthetic claim {i}",
        }
        for i in range(claims)
    ]
    edges: list[dict[str, str]] = [
        {"from": f"N{(i - 1) // 2}", "to": f"N{i}", "type": "supported_by"}
        for i in range(1, claims)
    ]
    solutions = 0
    for i in range(claims):
        if 2 * i + 1 < claims:
            continue
        name = f"a{i}.txt"
        payload = (f"synthetic artifact {i}\n" * 64)[:ARTIFACT_BYTES]
        with open(os.path.join(evidence_dir, name), "w", encoding="utf-8") as handle:
            handle.write(payload)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        solution_id = f"Sn{i}"
        nodes.append(
            {
                "id": solution_id,
                "type": "solution",
                "statement": f"synthetic solution {i}",
                "evidence": {"path": f"evidence/{name}", "sha256": digest},
            }
        )
        edges.append({"from": f"N{i}", "to": solution_id, "type": "supported_by"})
        solutions += 1
    document = {
        "name": f"synthetic {node_count}",
        "nodes": nodes,
        "edges": edges,
        "top_goals": ["N0"],
    }
    path = os.path.join(directory, f"case_{node_count}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return path, len(nodes), solutions


def main() -> int:
    start = time.perf_counter()
    lines = [
        "Run cost against case size -- assuregraph 0.1.0",
        "",
        "Two shared contended CPU cores. Wall clock, varies 10-20 % run to run.",
        f"Loader: PyYAML {LOADER_NAME} (libyaml present: {yaml.__with_libyaml__}).",
        "Claims form a complete binary tree; every claim leaf carries one 1 KiB",
        "evidence artifact whose digest matches. A correct run finds zero findings.",
        "Real assurance cases in the literature have tens to low hundreds of elements;",
        "the largest row is a stress case, not a representative one.",
        "",
        f"{'nodes':>7} {'edges':>7} {'solutions':>10} {'parse s':>9} {'checks s':>9} "
        f"{'mermaid s':>10} {'findings':>9} {'yaml KiB':>9}",
        "-" * 84,
    ]
    failures = 0
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for size in SIZES:
            directory = os.path.join(scratch, f"s{size}")
            os.makedirs(directory, exist_ok=True)
            path, node_count, solutions = build_case(directory, size)
            yaml_kib = os.path.getsize(path) / 1024

            t0 = time.perf_counter()
            case = load_case(path)
            parse_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            report = run_checks(case)
            check_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            diagram = render_mermaid(case, max_label_chars=40)
            mermaid_seconds = time.perf_counter() - t0

            # Expected: zero findings of any severity, at every size.
            if report.findings:
                failures += 1
            if len(case.nodes) != node_count:
                failures += 1
            if not diagram.startswith("flowchart"):
                failures += 1

            rows.append(
                (node_count, len(case.edges), solutions, parse_seconds, check_seconds,
                 mermaid_seconds, len(report.findings), yaml_kib, report.error_count)
            )
            lines.append(
                f"{node_count:>7} {len(case.edges):>7} {solutions:>10} {parse_seconds:>9.3f} "
                f"{check_seconds:>9.3f} {mermaid_seconds:>10.3f} "
                f"{len(report.findings):>9} {yaml_kib:>9.1f}"
            )

    lines.append("")
    lines.append("Findings per size (all must be 0 for a synthetic tree of fresh evidence):")
    for row in rows:
        lines.append(f"  {row[0]:>7} nodes -> {row[6]} findings, {row[8]} of them errors")
    biggest = rows[-1]
    lines.append("")
    lines.append("Per-node cost of run_checks (microseconds per node):")
    for row in rows:
        lines.append(f"  {row[0]:>7} nodes -> {1e6 * row[4] / row[0]:8.2f} us/node")
    # Compare the two largest rows rather than the smallest and the largest: at
    # 150 nodes the measurement is dominated by fixed interpreter overhead and
    # the ratio it produces moves by a factor of two between runs.
    reference = rows[2]
    node_ratio = biggest[0] / reference[0]
    check_ratio = biggest[4] / reference[4]
    lines.extend(
        [
            "",
            f"Growth from {reference[0]} to {biggest[0]} nodes",
            f"  node count ratio                 : {node_ratio:.1f}x",
            f"  run_checks wall-clock ratio      : {check_ratio:.1f}x",
            "  A ratio near the node ratio is consistent with the linear complexity the",
            "  modules document. It is consistent with it, not a proof of it: five sizes",
            "  on one contended machine cannot distinguish O(V+E) from O(V log V), and",
            "  the wall-clock figures move by 10-20 % between runs.",
            "",
            f"  largest case checked             : {biggest[0]} nodes, {biggest[1]} edges, "
            f"{biggest[2]} evidence artifacts",
            f"  its run_checks wall clock        : {biggest[4]:.3f} s",
            f"  its parse wall clock             : {biggest[3]:.3f} s",
            f"  its mermaid render wall clock    : {biggest[5]:.3f} s",
            "",
            f"FAILURES: {failures}",
            f"elapsed: {time.perf_counter() - start:.3f} s",
        ]
    )
    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
