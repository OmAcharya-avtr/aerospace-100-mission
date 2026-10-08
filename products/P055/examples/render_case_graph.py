"""Render the shipped cases as Mermaid, and draw the same graph as a PNG.

Writes, for the complete and the incomplete case:

* ``screenshots/<case>.mmd`` -- the Mermaid source, which is what a reviewer
  pastes into a GitHub README or pull request;
* ``screenshots/<case>_graph.png`` -- the same graph drawn directly, with GSN
  shapes, so the repository has a picture that needs no Mermaid renderer.

The PNG layout is a simple layered one: depth from the top goals by
breadth-first search sets the row, document order within a row sets the column.
It is not a GSN-standard layout engine and makes no attempt to minimise edge
crossings; a case of more than about thirty nodes should be read as Mermaid
rather than as this picture.

Run: ``python examples/render_case_graph.py``
"""

from __future__ import annotations

import os

import _bootstrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from assuregraph import (  # noqa: E402
    EdgeKind,
    NodeKind,
    Severity,
    load_case,
    node_depths,
    reachable_from,
    render_mermaid,
    run_checks,
)

KIND_COLOUR = {
    NodeKind.GOAL: "#dbe6f7",
    NodeKind.STRATEGY: "#e6dcf7",
    NodeKind.SOLUTION: "#d8f0e0",
    NodeKind.CONTEXT: "#ededed",
    NodeKind.ASSUMPTION: "#fdf2cf",
    NodeKind.JUSTIFICATION: "#fdf2cf",
}
BOX_WIDTH = 1.55
BOX_HEIGHT = 0.62


def _draw_node(axes, x, y, node, flagged):
    """Draw one GSN element at (x, y) with a shape that follows the standard."""
    edge_colour = "#b3261e" if flagged else "#333333"
    line_width = 2.2 if flagged else 0.9
    face = KIND_COLOUR[node.kind]
    half_w, half_h = BOX_WIDTH / 2, BOX_HEIGHT / 2
    if node.kind is NodeKind.GOAL:
        shape = patches.Rectangle((x - half_w, y - half_h), BOX_WIDTH, BOX_HEIGHT)
    elif node.kind is NodeKind.STRATEGY:
        skew = 0.28
        shape = patches.Polygon(
            [
                (x - half_w + skew, y - half_h),
                (x + half_w, y - half_h),
                (x + half_w - skew, y + half_h),
                (x - half_w, y + half_h),
            ],
            closed=True,
        )
    elif node.kind is NodeKind.SOLUTION:
        shape = patches.Ellipse((x, y), BOX_WIDTH * 0.78, BOX_HEIGHT * 1.25)
    elif node.kind is NodeKind.CONTEXT:
        shape = patches.FancyBboxPatch(
            (x - half_w + 0.12, y - half_h + 0.1),
            BOX_WIDTH - 0.24,
            BOX_HEIGHT - 0.2,
            boxstyle="round,pad=0.1,rounding_size=0.22",
        )
    else:  # Assumption and Justification: GSN draws an annotated ellipse.
        shape = patches.Ellipse((x, y), BOX_WIDTH * 0.95, BOX_HEIGHT * 1.1)
    shape.set_facecolor(face)
    shape.set_edgecolor(edge_colour)
    shape.set_linewidth(line_width)
    axes.add_patch(shape)

    marker = {NodeKind.ASSUMPTION: " (A)", NodeKind.JUSTIFICATION: " (J)"}.get(node.kind, "")
    suffix = "◇" if node.undeveloped else ""
    axes.text(
        x,
        y,
        f"{node.node_id}{marker}{suffix}",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold" if flagged else "normal",
    )


def _draw(case, report, title, target) -> None:
    depths = node_depths(case)
    reached = reachable_from(case, case.resolved_top_goals())
    unplaced_depth = (max(depths.values()) + 1) if depths else 0
    rows: dict[int, list[str]] = {}
    for node_id in case.nodes:
        depth = depths.get(node_id, unplaced_depth)
        rows.setdefault(depth, []).append(node_id)

    position: dict[str, tuple[float, float]] = {}
    widest = max(len(members) for members in rows.values())
    for depth, members in rows.items():
        for index, node_id in enumerate(members):
            x = (index - (len(members) - 1) / 2) * (BOX_WIDTH + 0.45)
            position[node_id] = (x, -depth * (BOX_HEIGHT + 0.95))

    flagged = {
        node_id
        for finding in report.findings
        if finding.severity is Severity.ERROR
        for node_id in finding.node_ids
    }

    figure, axes = plt.subplots(
        figsize=(max(8.0, widest * (BOX_WIDTH + 0.45) + 2.0), 1.6 + len(rows) * 1.45)
    )
    for edge in case.edges:
        x0, y0 = position[edge.source]
        x1, y1 = position[edge.target]
        supported = edge.kind is EdgeKind.SUPPORTED_BY
        if abs(y0 - y1) < 1e-9:
            # Same row (both nodes at the same depth, or both unplaced): leave
            # and enter from the sides, so the arrow does not loop below.
            side = 1.0 if x1 > x0 else -1.0
            tail = (x0 + side * BOX_WIDTH * 0.52, y0)
            head = (x1 - side * BOX_WIDTH * 0.52, y1)
        else:
            tail = (x0, y0 - BOX_HEIGHT * 0.62)
            head = (x1, y1 + BOX_HEIGHT * 0.62)
        axes.annotate(
            "",
            xy=head,
            xytext=tail,
            # GSN Standard v3 draws SupportedBy with a filled arrowhead and
            # InContextOf with a hollow one. Both are solid lines.
            arrowprops={
                "arrowstyle": "-|>",
                "facecolor": "#333333" if supported else "white",
                "edgecolor": "#333333",
                "linewidth": 1.0,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
                "mutation_scale": 12,
            },
        )
    for node_id, node in case.nodes.items():
        x, y = position[node_id]
        _draw_node(axes, x, y, node, node_id in flagged)
        if node_id not in reached:
            axes.text(
                x,
                y - BOX_HEIGHT * 0.95,
                "orphan",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#b3261e",
            )

    axes.set_title(title, fontsize=10)
    axes.set_aspect("equal")
    axes.autoscale_view()
    axes.margins(0.08, 0.12)
    axes.axis("off")
    legend = [
        patches.Patch(facecolor=KIND_COLOUR[kind], edgecolor="#333333", label=kind.value)
        for kind in NodeKind
    ]
    legend.append(
        patches.Patch(facecolor="white", edgecolor="#b3261e", linewidth=2.2, label="error finding")
    )
    axes.legend(handles=legend, loc="lower center", ncol=7, fontsize=7.5, frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def main() -> int:
    written: list[str] = []
    for filename in ("complete_case.yaml", "incomplete_case.yaml"):
        stem = filename.removesuffix(".yaml")
        case = load_case(os.path.join(_bootstrap.CASES, filename))
        report = run_checks(case)

        mermaid_path = os.path.join(_bootstrap.SCREENSHOTS, f"{stem}.mmd")
        with open(mermaid_path, "w", encoding="utf-8") as handle:
            handle.write(render_mermaid(case, report=report, max_label_chars=52, fence=True))
        written.append(mermaid_path)

        png_path = os.path.join(_bootstrap.SCREENSHOTS, f"{stem}_graph.png")
        _draw(
            case,
            report,
            f"{filename}  --  {len(case.nodes)} nodes, {len(case.edges)} relationships, "
            f"exit code {report.exit_code}\n"
            "shapes follow GSN Standard v3; a red outline marks a node named by an error "
            "finding; ◇ marks Undeveloped",
            png_path,
        )
        written.append(png_path)

    lines = ["wrote:"] + [
        f"  {os.path.relpath(path, _bootstrap.REPO_ROOT)}" for path in written
    ]
    lines.append("")
    for filename in ("complete_case.yaml", "incomplete_case.yaml"):
        case = load_case(os.path.join(_bootstrap.CASES, filename))
        report = run_checks(case)
        diagram = render_mermaid(case, report=report, max_label_chars=52)
        lines.append(f"{filename}: {len(diagram.splitlines())} lines of Mermaid, "
                     f"{len(case.nodes)} nodes, {len(case.edges)} edges, "
                     f"exit code {report.exit_code}")
    lines.append("")
    lines.append("first 12 lines of complete_case.mmd:")
    with open(os.path.join(_bootstrap.SCREENSHOTS, "complete_case.mmd"), encoding="utf-8") as fh:
        for line in fh.read().splitlines()[:12]:
            lines.append(f"  {line}")
    with open(
        os.path.join(_bootstrap.VALIDATION, "example_render_case_graph_output.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
