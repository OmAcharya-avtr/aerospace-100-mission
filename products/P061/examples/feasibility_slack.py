"""The deliverable as a picture: the binding constraint and every other slack.

One panel per shipped instance. Each bar is one declared constraint's normalised
slack; the binding one -- the least -- is drawn in black and labelled. The point
of the figure is that a different instance leaves a different constraint
binding, and the report says which without the reader having to know.

Run: ``python examples/feasibility_slack.py``
"""

from __future__ import annotations

from _common import EXACT, GREEDY, LEARNED, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.feasibility import evaluate  # noqa: E402
from activityplan.instances import INSTANCE_ORDER, RECOMMENDED_STEP_S, named_instance  # noqa: E402
from activityplan.timeline import RasterisedProblem  # noqa: E402


def main() -> int:
    style()
    fig, axes = plt.subplots(len(INSTANCE_ORDER), 1, figsize=(10.0, 13.0))
    summary = []
    for ax, name in zip(axes, INSTANCE_ORDER, strict=True):
        rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
        exact = solve_exact(rasterised, time_limit_s=150.0)
        report = evaluate(rasterised, exact.plan)
        entries = sorted(report.candidates, key=lambda entry: entry.normalised)
        labels = [entry.name for entry in entries]
        values = [min(entry.normalised, 2.0) for entry in entries]
        clipped = [entry.normalised > 2.0 for entry in entries]
        colours = [EXACT if index == 0 else GREEDY for index in range(len(entries))]
        bars = ax.barh(range(len(entries)), values, color=colours, height=0.65)
        for index, (bar, was_clipped) in enumerate(zip(bars, clipped, strict=True)):
            text = f"{entries[index].normalised:+.4f}"
            if was_clipped:
                text += "  (bar clipped at 2)"
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                text,
                va="center",
                fontsize=6.5,
                color=EXACT if index == 0 else GREEDY,
            )
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 2.45)
        ax.axvline(0.0, color=LEARNED, linewidth=0.8)
        ax.set_title(
            f"{name}: binding constraint {report.binding.name}, normalised slack "
            f"{report.binding.normalised:+.6f}; objective {exact.value:.1f} of "
            f"{rasterised.problem.total_priority:.1f}",
            fontsize=9,
        )
        ax.grid(axis="y", visible=False)
        summary.append((name, report.binding.name, report.binding.normalised, exact.value))
    axes[-1].set_xlabel("normalised slack = slack / declared normaliser (dimensionless)")
    fig.suptitle(
        "activityplan: the binding constraint is whichever normalised slack is least",
        fontsize=11,
        y=0.997,
    )
    fig.subplots_adjust(hspace=0.75, top=0.96)
    path = save(fig, "feasibility_slack.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    print(f"{'instance':18s} {'binding constraint':28s} {'normalised':>12s} {'objective':>10s}")
    for name, binding, normalised, value in summary:
        print(f"{name:18s} {binding:28s} {normalised:+12.6f} {value:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
