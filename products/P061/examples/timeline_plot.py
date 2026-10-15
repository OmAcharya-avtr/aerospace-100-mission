"""The flagship figure: a timeline with the resource traces that justify it.

Four stacked panels sharing one time axis for the ``leo-imaging`` instance:
the activity Gantt, the power balance, the battery state of charge against its
depth-of-discharge floor, and the recorder content against its capacity, with
the contact windows shaded on every panel.

Run: ``python examples/timeline_plot.py``
"""

from __future__ import annotations

import numpy as np
from _common import (
    BATTERY,
    EXACT,
    GREEDY,
    LOAD,
    RECORDER,
    SOLAR,
    add_src_to_path,
    save,
    style,
)

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.feasibility import evaluate  # noqa: E402
from activityplan.greedy import plan_greedy  # noqa: E402
from activityplan.instances import RECOMMENDED_STEP_S, named_instance  # noqa: E402
from activityplan.simulate import simulate  # noqa: E402
from activityplan.timeline import RasterisedProblem  # noqa: E402


def main() -> int:
    style()
    problem = named_instance("leo-imaging")
    step = RECOMMENDED_STEP_S["leo-imaging"]
    rasterised = RasterisedProblem.build(problem, step)
    greedy = plan_greedy(rasterised)
    exact = solve_exact(rasterised, time_limit_s=150.0)
    plan = exact.plan
    trace = simulate(rasterised, plan)
    report = evaluate(rasterised, plan)

    edges = rasterised.grid.edges_s() / 3600.0
    hours = problem.horizon_s / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 10.0), sharex=True)

    def shade_contacts(ax) -> None:
        for contact in problem.resources.contacts:
            ax.axvspan(
                contact.start_s / 3600.0,
                contact.end_s / 3600.0,
                color=SOLAR,
                alpha=0.14,
                linewidth=0,
            )

    ax = axes[0]
    names = [activity.name for activity in problem.activities]
    for row, name in enumerate(names):
        y = len(names) - 1 - row
        activity = problem.activity(name)
        latest = min(activity.latest_finish_s, problem.horizon_s)
        ax.hlines(
            y,
            activity.earliest_start_s / 3600.0,
            latest / 3600.0,
            color=GREEDY,
            linewidth=1.0,
            linestyle=":",
        )
        if name in plan.starts:
            start = plan.start_s(name) / 3600.0
            finish = rasterised.finish_s(problem.index(name), plan.starts[name]) / 3600.0
            ax.barh(
                y,
                finish - start,
                left=start,
                height=0.6,
                color=EXACT,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(
                finish + 0.06,
                y,
                f"p={activity.priority:.0f}",
                va="center",
                fontsize=7,
                color=EXACT,
            )
        else:
            ax.text(
                0.03,
                y,
                "not placed",
                va="center",
                fontsize=7,
                color=GREEDY,
                style="italic",
            )
    shade_contacts(ax)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=7)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_ylabel("activity")
    ax.set_title(
        f"leo-imaging, exact optimum: {exact.value:.0f} of "
        f"{problem.total_priority:.0f} priority units placed "
        f"(greedy baseline {greedy.value:.0f}). Dotted bars are declared windows; "
        f"shaded columns are contacts."
    )
    ax.grid(axis="x")

    ax = axes[1]
    solar_w = trace.solar_wh * 3600.0 / step
    load_w = trace.total_load_wh * 3600.0 / step
    ax.step(edges[:-1], solar_w, where="post", color=SOLAR, linewidth=1.4, label="array input")
    ax.step(edges[:-1], load_w, where="post", color=LOAD, linewidth=1.4, label="total load")
    ax.fill_between(
        edges[:-1],
        solar_w,
        load_w,
        where=solar_w >= load_w,
        step="post",
        color=SOLAR,
        alpha=0.22,
        linewidth=0,
        label="surplus",
    )
    ax.fill_between(
        edges[:-1],
        solar_w,
        load_w,
        where=solar_w < load_w,
        step="post",
        color=BATTERY,
        alpha=0.22,
        linewidth=0,
        label="deficit",
    )
    shade_contacts(ax)
    ax.set_ylabel("power [W]")
    ax.set_title("power balance, step averages. Deficit is what the battery has to cover.")
    ax.legend(ncol=4, loc="upper right", fontsize=7)

    ax = axes[2]
    ax.step(edges[1:], trace.battery_wh, where="pre", color=BATTERY, linewidth=1.6)
    ax.axhline(
        problem.resources.floor_wh,
        color=BATTERY,
        linestyle="--",
        linewidth=1.0,
        label=f"DoD floor {problem.resources.floor_wh:.1f} W*h",
    )
    ax.axhline(
        problem.resources.battery_capacity_wh,
        color=EXACT,
        linestyle=":",
        linewidth=1.0,
        label=f"capacity {problem.resources.battery_capacity_wh:.0f} W*h",
    )
    battery = report.slack("battery-depth-of-discharge")
    low = trace.min_battery_step
    ax.plot(
        [(low + 1) * step / 3600.0],
        [trace.min_battery_wh],
        marker="v",
        color=BATTERY,
        markersize=7,
    )
    near_right = (low + 1) * step > 0.7 * problem.horizon_s
    ax.annotate(
        f"min {trace.min_battery_wh:.2f} W*h, slack {battery.slack:.2f} W*h",
        xy=((low + 1) * step / 3600.0, trace.min_battery_wh),
        xytext=(-8 if near_right else 8, 12),
        textcoords="offset points",
        ha="right" if near_right else "left",
        fontsize=7,
        color=BATTERY,
    )
    shade_contacts(ax)
    ax.set_ylabel("battery [W*h]")
    ax.set_title(
        f"battery energy. {trace.shunted_wh:.1f} W*h shunted because the battery was full."
    )
    ax.legend(ncol=2, loc="lower left", fontsize=7)

    ax = axes[3]
    ax.step(edges[1:], trace.recorder_mbit, where="pre", color=RECORDER, linewidth=1.6)
    capacity = problem.resources.recorder_capacity_mbit
    ax.axhline(
        capacity,
        color=RECORDER,
        linestyle="--",
        linewidth=1.0,
        label=f"capacity {capacity:.0f} Mbit",
    )
    full = trace.max_recorder_step
    ax.plot([(full + 1) * step / 3600.0], [trace.max_recorder_mbit], marker="^", color=RECORDER)
    recorder = report.slack("recorder-capacity")
    ax.annotate(
        f"BINDING: slack {recorder.slack:.3f} Mbit",
        xy=((full + 1) * step / 3600.0, trace.max_recorder_mbit),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=8,
        color=RECORDER,
        weight="bold",
    )
    shade_contacts(ax)
    ax.set_ylabel("recorder [Mbit]")
    ax.set_xlabel("time from the start of the horizon [h]")
    ax.set_title(
        f"recorder content. Binding constraint of this plan: {report.binding.name}, "
        f"normalised slack {report.binding.normalised:+.6f}."
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0.0, hours)

    fig.suptitle(
        "activityplan: timeline and the resource traces that make it auditable",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.33, top=0.955)
    path = save(fig, "timeline_plot.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    print(f"exact objective     : {exact.value:.1f} of {problem.total_priority:.1f}")
    print(f"greedy objective    : {greedy.value:.1f}")
    print(f"binding constraint  : {report.binding.name}")
    print(
        f"                      slack {report.binding.slack:.6g} {report.binding.unit}, "
        f"normalised {report.binding.normalised:+.6f}"
    )
    print(f"activities placed   : {plan.n_placed} of {problem.n_activities}")
    print(f"not placed          : {', '.join(report.unplaced)}")
    print(f"battery minimum     : {trace.min_battery_wh:.4f} W*h against a floor of "
          f"{problem.resources.floor_wh:.4f} W*h")
    print(f"recorder maximum    : {trace.max_recorder_mbit:.4f} Mbit of {capacity:.1f} Mbit")
    print(f"energy shunted      : {trace.shunted_wh:.4f} W*h")
    print(f"data downlinked     : {trace.total_downlinked_mbit:.1f} Mbit of "
          f"{trace.total_produced_mbit:.1f} Mbit produced")
    residual = report.energy_audit()["residual"]
    print(f"energy residual     : {np.format_float_scientific(residual, 3)} W*h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
